function downloadSavedFile(content,type,name){
  const url=URL.createObjectURL(new Blob([content],{type}));
  const link=document.createElement('a');link.href=url;link.download=name;link.click();
  setTimeout(()=>URL.revokeObjectURL(url),1000);
}
function savedView(){
  const controls=[...document.querySelectorAll('aside input,aside select')].filter(input=>input.type!=='file');
  return {mode,stationsVisible,center:[map.getCenter().lat,map.getCenter().lng],zoom:map.getZoom(),
    controls:controls.map(input=>({id:input.id,parent:input.parentElement.closest('[id]')?.id||'',
      type:input.type,value:input.value,checked:Boolean(input.checked)})),
    details:[...document.querySelectorAll('aside details[id]')].map(item=>({id:item.id,open:item.open}))};
}
function validateSavedView(view){
  if(!view||!Array.isArray(view.center)||view.center.length!==2||
    !view.center.every(Number.isFinite)||Math.abs(view.center[0])>85||Math.abs(view.center[1])>180||
    !Number.isFinite(view.zoom)||view.zoom<3||view.zoom>15||typeof view.stationsVisible!=='boolean'||
    typeof view.mode!=='string'||!['uncertainty','baseline','reduction','affected','climate','placement'].includes(view.mode)&&
      !/^terrain:[a-z0-9_]+$/.test(view.mode)||
    !Array.isArray(view.controls)||view.controls.length>500||!Array.isArray(view.details)||view.details.length>100)
    throw new Error('Ugyldige kartinnstillinger i scenariofilen');
  for(const entry of view.controls)if(!entry||typeof entry.id!=='string'||typeof entry.parent!=='string'||
    typeof entry.value!=='string'||typeof entry.checked!=='boolean')throw new Error('Ugyldig panelinnstilling');
  for(const entry of view.details)if(!entry||typeof entry.id!=='string'||typeof entry.open!=='boolean')
    throw new Error('Ugyldig panelinnstilling');
}
async function saveEditableScenario(){
  if(busy||!state)return;
  if(!await commitPendingCountySelection())return;
  const view=savedView();busy=true;controls();
  try{
    const response=await fetch('/api/scenario'),document=await response.json();
    if(!response.ok)throw new Error(document.error||'Scenariet kunne ikke lagres');
    document.view=view;document.saved_at=new Date().toISOString();
    downloadSavedFile(JSON.stringify(document,null,2),'application/json','vaernett-scenario.json');
  }catch(failure){error(failure.message);}
  finally{busy=false;controls();}
}
async function loadEditableScenario(file){
  if(!file||busy)return;
  if(file.size>2*1024*1024){error('Scenariofilen er for stor');return;}
  try{
    const saved=JSON.parse(await file.text());validateSavedView(saved.view);
    if(!window.confirm('Åpne lagret scenario og erstatte det nåværende scenariet?'))return;
    if(saved.view.controls.some(item=>item.parent==='road-categories')&&!roadOptions){
      await initializeRoads();if(!roadOptions?.available)throw new Error('Vegnettet er ikke tilgjengelig');
    }
    if(saved.view.controls.some(item=>item.parent==='coverage-classes')&&!coverageOptions?.available){
      await initializeCoverage();
      if(!coverageOptions?.available)throw new Error('Dekningsgrad er ikke tilgjengelig');
    }
    busy=true;controls();clearPreview();
    const response=await fetch('/api/scenario',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(saved)});
    const result=await response.json();if(!response.ok)throw new Error(result.error||'Scenariet kunne ikke åpnes');
    state=result;mode=saved.view.mode;stationsVisible=saved.view.stationsVisible;render();
    for(const entry of saved.view.controls){
      const input=entry.id?element(entry.id):[...document.querySelectorAll('aside input')].find(input=>
        input.parentElement.closest('[id]')?.id===entry.parent&&input.value===entry.value);
      if(!input||!input.closest('aside')||input.type==='file')continue;
      if(input.type==='checkbox')input.checked=entry.checked;else input.value=entry.value;
    }
    if(!saved.view.controls.some(entry=>['road-score-4','road-score-5'].includes(entry.id))){
      const previous=saved.view.controls.find(entry=>entry.id==='road-scores');
      for(const id of ['road-score-4','road-score-5'])element(id).checked=previous?.checked??true;
    }
    for(const entry of saved.view.details){const details=element(entry.id);if(details?.matches('aside details'))details.open=entry.open;}
    for(const layer of [fixedMarkers,manualMarkers]){if(stationsVisible)layer.addTo(map);else map.removeLayer(layer);}
    render();populateTerrainOptions();map.setView(saved.view.center,saved.view.zoom);
    const visibility=element('station-visibility'),label=stationsVisible?'Skjul stasjoner':'Vis stasjoner';
    visibility.setAttribute('aria-pressed',String(stationsVisible));visibility.setAttribute('aria-label',label);
    visibility.title=label;visibility.textContent=label;
    updateMap();overlay?.setOpacity(overlayOpacity());scheduleRoads();scheduleKuldehull();scheduleCoverage();
  }catch(failure){error(failure.message);}
  finally{busy=false;controls();element('scenario-file').value='';}
}
async function imageData(url,signal){
  const response=await nativeFetch(url,{signal});if(!response.ok)throw new Error('Kartbildet kunne ikke lagres');
  const blob=await response.blob();
  return await new Promise((resolve,reject)=>{const reader=new FileReader();reader.onload=()=>resolve(reader.result);
    reader.onerror=()=>reject(new Error('Kartbildet kunne ikke leses'));reader.readAsDataURL(blob);});
}
const UNCERTAINTY_LEGEND={low:'Mindre usikkert',high:'Mer usikkert',
  gradient:'linear-gradient(to right,#053061,#6bacd1,#f7f7f7,#e48065,#67001f)'};
const DIRECT_P95_LEGEND_TITLE='Anslått 95 %-intervall (± °C)';
const ROAD_SCORE_COLOURS=['#00bcd4','#ffe119','#9e9600','#fe9a52','#e6007e'];
function displayRoadScoreClass(value){
  const scoreClass=Number(value);
  return scoreClass===5?1:scoreClass===4?2:value;
}
function emptyReductionLegendTitle(value,isP95){
  return `Anslått 95 %-intervall (± °C)`;
}
function weatherLegendMarkup(title,fixedP95=false){
  const escape=value=>String(value).replace(/[&<>"']/g,character=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[character]));
  return '<div class="legend"><div class="legend-title">'+escape(title)+'</div>'+
    '<div class="ramp" style="background:'+UNCERTAINTY_LEGEND.gradient+'"></div><div class="ticks"><span>'+
    (fixedP95?'0 °C</span><span style="position:absolute;left:50%;transform:translateX(-50%)">1 °C</span><span>≥3 °C':
      UNCERTAINTY_LEGEND.low+'</span><span>'+UNCERTAINTY_LEGEND.high)+'</span></div></div>';
}
async function compressedJson(value){
  if(typeof CompressionStream==='undefined')throw new Error('HTML-eksport krever en oppdatert nettleser med gzip-støtte');
  const stream=new Blob([JSON.stringify(value)]).stream().pipeThrough(new CompressionStream('gzip'));
  const bytes=new Uint8Array(await new Response(stream).arrayBuffer()),parts=[];
  for(let offset=0;offset<bytes.length;offset+=32768)parts.push(String.fromCharCode(...bytes.subarray(offset,offset+32768)));
  return btoa(parts.join(''));
}
async function frozenMapRuntime(){
  const data=JSON.parse(document.getElementById('snapshot').textContent);
  const map=L.map('map',{minZoom:3,maxZoom:15}).setView(data.center,data.zoom);
  for(const [name,z] of [['weather',400],['background-roads',420],['coverage',425],['roads',430],['basins',435],['stations',450],['calculation-points',445],['county-borders',440]])
    map.createPane(name).style.zIndex=z;
  map.getPane('county-borders').style.pointerEvents='none';
  map.getPane('calculation-points').style.pointerEvents='none';
  const note=document.getElementById('saved-note');
  let backgroundFailed=false;
  const backgrounds={
    'Topografisk bakgrunn':L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}',{
      attribution:'Tiles &copy; Esri &mdash; Sources: Esri, HERE, Garmin, Intermap, increment P Corp., GEBCO, USGS, FAO, NPS, NRCAN, GeoBase, IGN, Kadaster NL, Ordnance Survey, Esri Japan, METI, Esri China (Hong Kong), OpenStreetMap contributors, GIS User Community',maxZoom:19})
  };
  backgrounds['Topografisk bakgrunn'].on('tileerror',()=>{
    backgroundFailed=true;note.hidden=false;note.textContent='Bakgrunnskartet kunne ikke lastes. Kontroller internettilkoblingen.';
  }).on('load',()=>{if(backgroundFailed){backgroundFailed=false;note.hidden=true;}});
  backgrounds['Topografisk bakgrunn'].addTo(map);
  const weather=L.layerGroup(),baselineWeather=L.layerGroup(),reductionWeather=L.layerGroup(),placementWeather=L.layerGroup(),backgroundOnly=L.layerGroup(),coverage=L.layerGroup(),
    basins=L.layerGroup(),stations=L.layerGroup(),roads=L.layerGroup(),counties=L.layerGroup(),debug=L.layerGroup();
  const roadOverview=L.layerGroup(),backgroundRoadOverview=L.layerGroup(),basinOverview=L.layerGroup();
  L.imageOverlay(data.basinOverview.url,data.basinOverview.bounds,{opacity:.75,pane:'basins',interactive:false}).addTo(basinOverview);
  const roadOnly=document.getElementById('saved-road-only');
  roadOnly.checked=data.weatherExtent==='roads';
  let activeWeather=data.full.weather.find(item=>item.id===(roadOnly.checked?'roads':'counties'));
  const popupBox=(title,rows)=>{
    const box=document.createElement('div'),heading=document.createElement('strong');
    heading.textContent=title;box.append(heading);
    for(const [label,value] of rows){
      if(value===undefined||value===null||value==='')continue;
      const row=document.createElement('div');row.textContent=label?label+': '+value:String(value);box.append(row);
    }
    return box;
  };
  const propertyLabels={roadid:'Veg',streetname:'Vegnavn',Vegscore:'Vegscore',VS_Total:'Vegscore',VS_Total_klasse:'Vegscoreklasse',
    length_m:'Lengde (m)',mean_error:'Gjennomsnittlig modellfeil',
    max_error:'Største modellfeil',municipality:'Kommune',
    links:'Antall lenker',overview:'Forenklet oversikt'};
  const unpack=async packed=>{
    if(typeof DecompressionStream==='undefined')throw new Error('Nettleseren støtter ikke komprimerte kartdata. Bruk en oppdatert Chrome, Edge eller Firefox.');
    const bytes=Uint8Array.from(atob(packed),character=>character.charCodeAt(0));
    const stream=new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
    return new Response(stream).json();
  };
  const vectors=[...data.vectors,...(data.roadPreview?await unpack(data.roadPreview):[])];
  for(const item of vectors){
    const group=item.pane==='roads'?roadOverview:item.pane==='background-roads'?backgroundRoadOverview:
      item.pane==='county-borders'?counties:item.pane==='stations'?stations:debug;
    L.geoJSON(item.geojson,{pane:item.pane||'overlayPane',style:item.style,
      interactive:item.pane==='roads'&&data.roadScores!==false,bubblingMouseEvents:false,
      pointToLayer:(f,p)=>L.circleMarker(p,{...item.style,pane:item.pane||'overlayPane'}),
      onEachFeature:(feature,layer)=>{
        const properties=feature.properties||{};
        const rows=Object.entries(propertyLabels).filter(([key])=>properties[key]!=null).map(([key,label])=>
          [label,key==='VS_Total_klasse'?displayRoadScoreClass(properties[key]):
            typeof properties[key]==='boolean'?(properties[key]?'Ja':'Nei'):
            typeof properties[key]==='object'?JSON.stringify(properties[key]):typeof properties[key]==='number'?Math.round(properties[key]):properties[key]]);
        if(rows.length)layer.bindPopup(popupBox('Veglenke',rows));
      }}).addTo(group);
  }
  for(const item of data.markers){
    const info=item.info||{},rows=[];
    rows.push(['Vegværstasjon',info.svv?'Ja':'Nei'],
      ['Eier',(info.owners||[]).join(', ')||'Ikke oppgitt'],
      ['Status',item.original?'Opprinnelig plassering':info.removed?'Fjernet fra scenariet':info.original_position?'Flyttet i scenariet':info.sensors_changed||info.added_sensors?.length?'Sensorer endret':'']);

    for(const sensor of info.removed_sensor_labels||[])rows.push(['Fjernet sensor (scenario)',sensor]);
    const box=popupBox(item.title||'Kartmarkør',rows);
    for(const sensor of item.sensors){
      const row=document.createElement('div'),dot=document.createElement('span');
      dot.className='sensor-dot';dot.style.background=sensor.color;
      row.append(dot,document.createTextNode(sensor.label));box.append(row);
    }
    const marker=L.marker(item.position,{pane:item.pane||'markerPane',zIndexOffset:item.zIndexOffset||0,
      interactive:item.interactive!==false,bubblingMouseEvents:false,
      title:item.title,savedStation:info,
      icon:L.divIcon({html:item.html,className:item.className,iconSize:item.size,iconAnchor:item.anchor})})
      .bindPopup(box,{maxWidth:360,maxHeight:350});
    marker.addTo(item.pane==='calculation-points'?debug:stations);
  }
  const climate=L.imageOverlay(data.climate.url,data.climate.bounds,{opacity:.75,pane:'weather',interactive:false});
  const overlays={'Værstasjoner':stations,'Vegnett':roads,'Kuldehull':basins};
  const coverageOverview=L.layerGroup();
  if(data.coverageOverview){
    const image=data.coverageOverview;
    L.imageOverlay(image.url,image.bounds,{opacity:.85,interactive:false,pane:'coverage'}).addTo(coverageOverview);
    overlays['Dekningsgrad']=coverage;
    if(data.coverageVisible)coverage.addTo(map);
  }
  debug.addTo(stations);
  counties.addTo(map);
  (data.mode==='climate'?climate:data.mode==='baseline'?baselineWeather:data.mode==='reduction'?reductionWeather:data.mode==='placement'?placementWeather:weather).addTo(map);
  counties.addTo(map);
  if(data.roadScores||data.roadsVisible)roads.addTo(map);
  if(data.roadsVisible)backgroundRoadOverview.addTo(map);
  if(data.stationsVisible)stations.addTo(map);
  if(data.basinsVisible)basins.addTo(map);
  L.control.layers({'Usikkerhet':weather,'Grunnett':baselineWeather,'Endring i usikkerhet':reductionWeather,'Plasseringsgevinst':placementWeather,'Vegklimasonekart':climate,
    'Bare bakgrunnskart':backgroundOnly},overlays,{collapsed:false,position:'topright'}).addTo(map);
  L.control.scale({imperial:false}).addTo(map);
  const legend=document.getElementById('legend-container');
  const updateLegend=()=>{
    legend.replaceChildren();
    if(map.hasLayer(weather)||map.hasLayer(baselineWeather)){
      const fixedP95=activeWeather.dataset==='temperature:direct_p95';
      legend.innerHTML=weatherLegendMarkup(fixedP95?DIRECT_P95_LEGEND_TITLE:'Temperatur',fixedP95);
    }
    if(map.hasLayer(reductionWeather)){
      legend.innerHTML=weatherLegendMarkup(activeWeather.empty_reference_rmse?
        emptyReductionLegendTitle(activeWeather.empty_reference_rmse,activeWeather.dataset==='temperature:direct_p95'):
        'Endringer fra opprinnelig grunnnett');
    }
    if(map.hasLayer(placementWeather)){
      const box=document.createElement('div');box.className='legend';
      const heading=document.createElement('div');heading.className='legend-title';heading.textContent='Plasseringsgevinst';
      if(data.full.placement.empty_reference_rmse)heading.textContent+=` · lagret referanse ${data.full.placement.empty_reference_rmse.toLocaleString('nb-NO')} °C`;
      const ramp=document.createElement('div');ramp.className='ramp';
      ramp.style.background='linear-gradient(to right,#ffffcc,#ffeda0,#feb24c,#f03b20,#800026)';
      const ticks=document.createElement('div');ticks.className='ticks';
      for(const text of ['Mindre gevinst','Mer gevinst']){const item=document.createElement('span');item.textContent=text;ticks.append(item);}
      box.append(heading,ramp,ticks);legend.append(box);
    }
    if(map.hasLayer(climate)){
      const box=document.createElement('div');box.className='legend climate-snapshot-legend';
      const heading=document.createElement('strong');heading.textContent='Vegklimasonekart';box.append(heading);
      for(const zone of data.climate.zones){
        const row=document.createElement('div'),swatch=document.createElement('span');
        swatch.style.cssText='display:inline-block;width:12px;height:12px;margin-right:6px;background:'+zone.color;
        row.append(swatch,document.createTextNode(zone.name));box.append(row);
      }
      legend.append(box);
    }
    if(map.hasLayer(coverage)){
      const box=popupBox('Dekningsgrad',[]);box.className='legend climate-snapshot-legend';
      for(const item of data.coverageClasses){
        const row=document.createElement('div'),dot=document.createElement('span');
        dot.className='sensor-dot';dot.style.background=item.color;
        row.append(dot,document.createTextNode(item.label));box.append(row);
      }
      legend.append(box);
    }
  };
  const opacity=document.getElementById('saved-opacity');
  opacity.value=data.opacity;
  opacity.oninput=()=>{
    const value=1-Number(opacity.value)/100;
    for(const group of [weather,baselineWeather,reductionWeather,placementWeather])group.eachLayer(layer=>layer.setOpacity(value));
    climate.setOpacity(value);
  };
  const selectWeather=()=>{
    activeWeather=data.full.weather.find(item=>item.id===(roadOnly.checked?'roads':'counties'));
    for(const [group,field] of [[weather,'url'],[baselineWeather,'baseline'],[reductionWeather,'reduction']]){
      group.clearLayers();
      L.imageOverlay(activeWeather[field],activeWeather.bounds,{
        opacity:1-Number(opacity.value)/100,pane:'weather',interactive:false}).addTo(group);
    }
    placementWeather.clearLayers();
    L.imageOverlay(roadOnly.checked?data.full.placement.roads_url:data.full.placement.url,
      data.full.placement.bounds,{opacity:1-Number(opacity.value)/100,pane:'weather',interactive:false}).addTo(placementWeather);
    stations.eachLayer(layer=>{
      if(!(layer instanceof L.Marker))return;
      const info=layer.options.savedStation;
      if(!info)return;
      const sensors=[...(info.sensors||[]),...(info.added_sensors||[])];
      layer.getElement()?.classList.toggle('inactive-sensors',!sensors.includes('temperature'));
    });
    updateLegend();
  };
  roadOnly.onchange=selectWeather;
  map.on('baselayerchange overlayadd overlayremove',updateLegend);
  map.on('layeradd',event=>{if(event.layer===stations)selectWeather();});
  selectWeather();
  const detailStatus=document.getElementById('detail-status');
  // One canvas hit-tests both layers; separate canvases block clicks through empty pixels.
  const detailRenderer=L.canvas({padding:.15,pane:'basins'});
  const backgroundRoadRenderer=data.roadScores===false?L.canvas({padding:.15,pane:'roads'}):detailRenderer;
  let detailSerial=0;
  const decoded=new Map();
  const intersects=(bbox,bounds)=>bbox[0]<=bounds.getEast()&&bbox[2]>=bounds.getWest()&&bbox[1]<=bounds.getNorth()&&bbox[3]>=bounds.getSouth();
  const decode=async chunk=>{
    if(decoded.has(chunk)){const result=decoded.get(chunk);decoded.delete(chunk);decoded.set(chunk,result);return result;}
    const result=await unpack(chunk.data);
    decoded.set(chunk,result);
    if(decoded.size>24)decoded.delete(decoded.keys().next().value);
    return result;
  };
  const refreshDetails=async()=>{
    const serial=++detailSerial,view=map.getBounds();
    detailStatus.textContent='';
    roads.clearLayers();basins.clearLayers();coverage.clearLayers();
    const visible=[['coverage',coverage],['basins',basins],['roads',roads]].filter(([,group])=>map.hasLayer(group));
    if(!visible.length){detailStatus.textContent='';return;}
    if(map.getZoom()<10){
      roadOverview.addTo(roads);basinOverview.addTo(basins);coverageOverview.addTo(coverage);
      return;
    }
    try{
      let total=0;
      for(const [kind,group] of visible){
        const chunks=data.full[kind].chunks.filter(chunk=>intersects(chunk.bounds,view));
        for(const chunk of chunks){
          const collection=await decode(chunk);
          if(serial!==detailSerial)return;
          const features=collection.features.filter(feature=>intersects(feature.bbox,view));
          total+=features.length;
          if(total>30000){
            roads.clearLayers();basins.clearLayers();coverage.clearLayers();
            detailStatus.textContent='Zoom nærmere for å vise alle detaljene i utsnittet.';
            return;
          }
          L.geoJSON({type:'FeatureCollection',features},{
            renderer:kind==='roads'?backgroundRoadRenderer:detailRenderer,bubblingMouseEvents:false,
            pane:kind==='roads'&&data.roadScores===false?'roads':'basins',
            interactive:kind!=='roads'||data.roadScores!==false,
            style:feature=>kind==='coverage'?{stroke:false,
              fillColor:data.coverageClasses.find(item=>item.value===feature.properties.value).color,fillOpacity:.85}:
              kind==='roads'?(data.roadScores===false?{color:'#000',weight:1.5,opacity:.9}:
                {color:ROAD_SCORE_COLOURS[Number(feature.properties.VS_Total_klasse)-1]||'#7b8480',weight:2.5,opacity:.95}):
              {color:'#00a6ca',weight:1,fillColor:'#00bce4',fillOpacity:.25},
            onEachFeature:(feature,layer)=>{
              const p=feature.properties;
              if(kind==='coverage'){
                layer.bindPopup(popupBox('Dekningsgrad',[['',data.coverageClasses.find(item=>item.value===p.value).label]]));
                return;
              }
              const fields=kind==='roads'?propertyLabels:{max_depth_m:'Maksdybde (m)',mean_depth_m:'Middeldybde (m)',
                area_m2:'Areal (m²)',volume_m3:'Volum (m³)'};
              const rows=Object.entries(fields).filter(([key])=>p[key]!=null).map(([key,label])=>
                [label,typeof p[key]==='number'?p[key].toLocaleString('nb-NO',{maximumFractionDigits:2}):p[key]]);
              layer.bindPopup(popupBox(kind==='roads'?'Veglenke':'Kuldehull',rows));
            }}).addTo(group);
          await new Promise(resolve=>setTimeout(resolve,0));
          if(serial!==detailSerial)return;
        }
      }
    }catch(failure){
      if(serial===detailSerial){roads.clearLayers();basins.clearLayers();coverage.clearLayers();detailStatus.textContent=failure.message;}
    }
  };
  let detailTimer=null;
  map.on('moveend overlayadd overlayremove',()=>{clearTimeout(detailTimer);detailSerial++;detailTimer=setTimeout(refreshDetails,100);});
  await refreshDetails();
  const canvas=document.createElement('canvas'),context=canvas.getContext('2d',{willReadFrequently:true});
  const climateImage=new Image();
  climateImage.onload=()=>{canvas.width=climateImage.naturalWidth;canvas.height=climateImage.naturalHeight;if(context)context.drawImage(climateImage,0,0);};
  climateImage.src=data.climate.url;
  const climateZone=point=>{
    if(!canvas.width||!context)return null;
    const bounds=L.latLngBounds(data.climate.bounds);
    if(!bounds.contains(point))return null;
    const northwest=L.CRS.EPSG3857.project(bounds.getNorthWest()),southeast=L.CRS.EPSG3857.project(bounds.getSouthEast());
    const projected=L.CRS.EPSG3857.project(point);
    const x=Math.floor((projected.x-northwest.x)/(southeast.x-northwest.x)*canvas.width);
    const y=Math.floor((northwest.y-projected.y)/(northwest.y-southeast.y)*canvas.height);
    if(x<0||y<0||x>=canvas.width||y>=canvas.height)return null;
    const pixel=context.getImageData(x,y,1,1).data;
    if(!pixel[3])return null;
    const color='#'+[...pixel].slice(0,3).map(value=>value.toString(16).padStart(2,'0')).join('');
    return data.climate.zones.find(zone=>zone.color.toLowerCase()===color)||null;
  };
  map.on('click',event=>{
    const rows=[],zone=climateZone(event.latlng);
    let title=zone?zone.name:'Dette området';
    if(!map.hasLayer(climate)&&zone)rows.push(['Klimasone',zone.name]);
    if(map.hasLayer(climate)){
      title=zone?zone.name:'Ingen klimasone i det lagrede kartbildet';
      const fields=[['vintertemp','Vintertemperatur (°C)'],['sommertemp','Sommertemperatur (°C)'],
        ['vinternedbor','Vinternedbør (mm)'],['frostdager','Frostdager']];
      const statistics=data.climate.statistics?.[zone?.value]||{};
      if(zone)rows.push(['','Gjennomsnitt for klimasonen:']);
      for(const [field,label] of fields)if(Number.isFinite(statistics[field]))
        rows.push([label,statistics[field].toLocaleString('nb-NO',{maximumFractionDigits:2})]);
      
    }
    if(map.hasLayer(stations)&&data.markers.length){
      const nearest=data.markers.filter(item=>item.info?.station_id).reduce((best,item)=>{
        const distance=event.latlng.distanceTo(L.latLng(item.position));
        return !best||distance<best.distance?{item,distance}:best;
      },null);
      rows.push(['Nærmeste værstasjon',nearest.item.title],['Avstand i luftlinje',(nearest.distance/1000).toFixed(2)+' km']);
    }
    if(!rows.length)rows.push(['','Velg Klimasoner eller vis værstasjonene for mer informasjon om området.']);
    L.popup({maxWidth:360,maxHeight:350}).setLatLng(event.latlng).setContent(popupBox(title,rows)).openOn(map);
  });
}
async function saveFrozenMap(){
  if(busy||!state)return;
  if(!await commitPendingCountySelection())return;
  updateMap();
  const pending=[...loadingTasks.values()].some(label=>label!=='Laster bakgrunnskart');
  const visibleImages=[];map.eachLayer(layer=>{if(layer instanceof L.ImageOverlay)visibleImages.push(layer);});
  if(pending||roadBusy||roadPending||
    element('coverage-show')?.checked&&coverageClasses().length&&(!coverageOverlay||element('coverage-status').textContent)||
    element('kuldehull-show').checked&&element('kuldehull-status').textContent||
    visibleImages.some(layer=>!layer.getElement()?.complete||!layer.getElement()?.naturalWidth)){
    error('Vent til kartlagene er ferdig lastet før du lagrer');return;
  }
  busy=true;controls();
  try{
    if(!roadOptions){
      if(!roadInitialization)roadInitialization=initializeRoads().finally(()=>{roadInitialization=null;});
      await roadInitialization;
    }
    if(!getRoadSettings(true))throw new Error('Vegfiltrene kunne ikke lastes');
    const viewKey=JSON.stringify(savedView());
    const snapshot={center:[map.getCenter().lat,map.getCenter().lng],zoom:map.getZoom(),
      title:document.title,saved:new Date().toISOString(),vectors:[],markers:[],
      mode:['climate','baseline','reduction','placement'].includes(mode)?mode:'uncertainty',
      weatherExtent:element('road-corridor').checked?'roads':'counties',
      opacity:element('opacity').value,stationsVisible,basinsVisible:element('kuldehull-show').checked,
      dataset:state.dataset,roadsVisible:element('road-show').checked,
      roadScores:selectedRoadScoreClasses().length>0,stationOwners:state.station_owners};
    const revision=state.revision;
    const layers=[];map.eachLayer(layer=>layers.push(layer));
    for(const group of [fixedMarkers,manualMarkers])if(!map.hasLayer(group))group.eachLayer(layer=>layers.push(layer));
    const exportPoints=L.layerGroup();buildCalculationPoints(exportPoints,true);
    const exportLayers=layers.filter(layer=>layer.options?.pane!=='calculation-points'&&layer!==kuldehullOverlay&&!coverageImages.has(layer));
    exportPoints.eachLayer(layer=>exportLayers.push(layer));
    for(const layer of exportLayers){
      if(layer instanceof L.Marker){
        const icon=layer.options.icon,point=layer.getLatLng();
        if(typeof icon?.options?.html!=='string')throw new Error('En kartmarkør kan ikke lagres');
        const station=layer.options.savedStation||[...state.fixed_stations,...state.stations].find(s=>
          Math.abs(s.lon-point.lng)<1e-9&&Math.abs(s.lat-point.lat)<1e-9||
          s.original_position&&Math.abs(s.original_position.lon-point.lng)<1e-9&&Math.abs(s.original_position.lat-point.lat)<1e-9);
        const proposal=station&&sensorProposal(station);
        const sensors=[...new Set([...(station?.sensors||[]),...(proposal?.sensors||[])])];
        snapshot.markers.push({position:[point.lat,point.lng],html:icon.options.html,className:icon.options.className,
          size:icon.options.iconSize,anchor:icon.options.iconAnchor,
          title:station?`${station.station_id||'Eget forslag'}${station.name?' · '+station.name:''}`:layer.options.title||'',
          pane:layer.options.pane,zIndexOffset:layer.options.zIndexOffset,
          interactive:layer.options.interactive,original:layer.options.originalStationPosition||
            Boolean(station?.original_position&&Math.abs(station.original_position.lon-point.lng)<1e-9&&Math.abs(station.original_position.lat-point.lat)<1e-9),
          info:station?{...station,added_sensors:proposal?.sensors||[],
            removed_sensor_labels:((state.frost_edits||[]).find(edit=>edit.station_id===station.station_id)?.removed_sensors||[])
              .map(sensor=>SENSORS[sensor]?.[0]||sensor)}:null,
          sensors:sensors.map(sensor=>({label:(SENSORS[sensor]?.[0]||sensor)+
            (proposal?.sensors.includes(sensor)?' · Lagt til (forslag)':''),color:SENSORS[sensor]?.[1]||'#929995'}))});
      }else if(layer instanceof L.Path&&layer.toGeoJSON){
        const style={};for(const key of ['color','weight','opacity','fill','fillColor','fillOpacity','dashArray','radius'])
          if(layer.options[key]!==undefined)style[key]=layer.options[key];
        snapshot.vectors.push({geojson:layer.toGeoJSON(),style,pane:layer.options.pane});
      }
    }
    snapshot.coverageClasses=coverageOptions?.classes.filter(item=>coverageClasses().includes(item.value))||[];
    snapshot.coverageVisible=Boolean(element('coverage-show').checked);
    const climateResponse=await fetch('/api/climate/options');
    const climate=await climateResponse.json();
    if(!climateResponse.ok)throw new Error(climate.error||'Klimasonene kunne ikke lagres');
    snapshot.climate={...climate,url:await imageData('/api/climate.png?size=1600')};
    const basinBounds=state.bounds;
    const basinQuery=new URLSearchParams({bounds:JSON.stringify([basinBounds[0][1],basinBounds[0][0],basinBounds[1][1],basinBounds[1][0]]),
      depth:element('kuldehull-depth').value,width:800,height:1600});
    snapshot.basinOverview={url:await imageData(`/api/kuldehull.png?${basinQuery}`),bounds:basinBounds};
    if(snapshot.coverageClasses.length){
      const query=new URLSearchParams({bounds:basinQuery.get('bounds'),width:1600,height:1600,
        classes:JSON.stringify(snapshot.coverageClasses.map(item=>item.value))});
      snapshot.coverageOverview={url:await imageData(`/api/coverage.png?${query}`),bounds:basinBounds};
    }
    const roadPreview=snapshot.vectors.filter(item=>item.pane==='roads');
    snapshot.roadPreview=await compressedJson(roadPreview);
    snapshot.vectors=snapshot.vectors.filter(item=>item.pane!=='roads');
    const fullResponse=await fetch('/api/snapshot-data',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({revision,roads:getRoadSettings(true),roads_visible:snapshot.roadsVisible,road_scores:snapshot.roadScores,
        weather_roads:getWeatherRoadSettings(true),include_placement:true,depth:Number(element('kuldehull-depth').value),
        coverage_classes:snapshot.coverageClasses.map(item=>item.value)})});
    snapshot.full=await fullResponse.json();
    if(!fullResponse.ok)throw new Error(snapshot.full.error||'Kartdataene kunne ikke lagres');
    if(snapshot.full.road_overview)snapshot.roadPreview=await compressedJson([...roadPreview,{
      geojson:snapshot.full.road_overview,pane:'background-roads',style:{color:'#000',weight:1.5,opacity:.9}
    }]);
    if(snapshot.full.revision!==revision||state.revision!==revision)throw new Error('Scenariet ble endret under lagring. Prøv igjen.');
    if(viewKey!==JSON.stringify(savedView()))throw new Error('Kartvisningen ble endret under lagring. Prøv igjen.');
    const corridor=snapshot.full.weather.find(item=>item.id==='roads');
    if(!corridor)throw new Error('Usikkerhetskart langs veger mangler i eksporten');
    const corridorLabel=Number(corridor.distance_m).toLocaleString('nb-NO')+' m';
    const styles=[...document.querySelectorAll('style')].map(style=>style.textContent).join('\n');
    const payload=JSON.stringify(snapshot).replace(/</g,'\\u003c');
    const runtime='const UNCERTAINTY_LEGEND='+JSON.stringify(UNCERTAINTY_LEGEND)+';'+
      'const DIRECT_P95_LEGEND_TITLE='+JSON.stringify(DIRECT_P95_LEGEND_TITLE)+';'+
      'const ROAD_SCORE_COLOURS='+JSON.stringify(ROAD_SCORE_COLOURS)+';'+displayRoadScoreClass.toString()+';'+
      emptyReductionLegendTitle.toString()+';'+weatherLegendMarkup.toString()+';'+
      '('+frozenMapRuntime.toString()+')().catch(failure=>{const note=document.getElementById("saved-note");note.hidden=false;note.textContent=failure.message;});';
    const snapshotStyle=`
      body{display:block;margin:0}
      #map{position:fixed;inset:0;width:100%;height:100%}
      .leaflet-tile-pane{filter:none}
      #saved-tools{position:fixed;top:10px;left:55px;z-index:1000;background:white;padding:8px;border-radius:4px}
      #saved-tools label{display:block;font-size:12px}
      #detail-status{max-width:330px;font-size:12px;margin-top:6px}
      #saved-owners{max-height:35vh;max-width:380px;overflow:auto}
      #legend-container{position:fixed;bottom:25px;left:10px;z-index:1000;max-height:45vh;overflow:auto}
      #legend-container .legend{position:static;inset:auto;width:auto;max-width:300px;background:white;padding:12px;margin:8px 0 0}
      .climate-snapshot-legend{font-size:11px}
      .leaflet-control-layers{max-height:60vh;overflow:auto}
      .leaflet-popup-content{font-size:12px;line-height:1.6}
      #saved-note{position:fixed;bottom:0;left:0;right:0;z-index:1100;background:white;padding:8px;font-size:12px}
      .sensor-dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px}
      @media(max-width:650px){
        #saved-tools{left:10px;top:90px}
        #legend-container{max-height:25vh;bottom:65px}
      }`;
    const html='<!doctype html><html lang="nb"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'+
      '<title>Værnett</title><link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"><style>'+
      styles+'</style><style>'+snapshotStyle+'</style></head><body>'+
      '<div id="map"></div><div id="saved-note" role="status" hidden></div><div id="saved-tools"><label><input id="saved-road-only" type="checkbox"> Vis usikkerhet langs veg</label><label>Viser usikkerhet inntil '+corridorLabel+' fra veg</label><label>Kartlagets gjennomsiktighet <input id="saved-opacity" type="range" min="0" max="100" step="5"></label><div id="detail-status" role="status"></div></div><div id="legend-container"></div><script type="application/json" id="snapshot">'+
      payload+'</script><script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script><script>'+runtime+'</script></body></html>';
    downloadSavedFile(html,'text/html','vaernett-kart.html');
  }catch(failure){error(failure.message);}
  finally{busy=false;controls();}
}
let coverageOptions=null,coverageOverlay=null,coverageController=null,coverageTimer=null,coverageKey=null,coverageSerial=0;
const coverageImages=new Set();
function coverageClasses(){
  return [...element('coverage-classes').querySelectorAll('input:checked')].map(input=>Number(input.value));
}
function scheduleCoverage(){
  if(!coverageOptions?.available||!state){clearTimeout(coverageTimer);return;}
  const classes=coverageClasses();
  if(!element('coverage-show').checked||!classes.length){
    clearTimeout(coverageTimer);
    coverageSerial++;coverageController?.abort();coverageKey=null;
    for(const image of coverageImages)map.removeLayer(image);coverageImages.clear();
    coverageOverlay=null;element('coverage-status').textContent='';return;
  }
  const settings=getKuldehullSettings();
  const key=JSON.stringify([settings.bounds,settings.width,settings.height,classes,state.counties]);
  if(key===coverageKey)return;
  clearTimeout(coverageTimer);
  coverageKey=key;
  const serial=++coverageSerial;
  coverageController?.abort();coverageController=new AbortController();
  const controller=coverageController;
  for(const image of coverageImages)map.removeLayer(image);coverageImages.clear();coverageOverlay=null;
  coverageTimer=setTimeout(async()=>{
    element('coverage-status').textContent='Laster dekningsgrad…';
    const query=new URLSearchParams({bounds:JSON.stringify(settings.bounds),width:settings.width,
      height:settings.height,classes:JSON.stringify(classes)});
    try{
      const url=await imageData(`/api/coverage.png?${query}`,controller.signal);
      if(serial!==coverageSerial)return;
      const next=L.imageOverlay(url,[[settings.bounds[1],settings.bounds[0]],[settings.bounds[3],settings.bounds[2]]],
        {opacity:.85,interactive:false,zIndex:425}).addTo(map);
      coverageImages.add(next);
      next.once('load',()=>{
        if(serial!==coverageSerial){map.removeLayer(next);coverageImages.delete(next);return;}
        coverageOverlay=next;element('coverage-status').textContent='';
      });
      next.once('error',()=>{
        map.removeLayer(next);coverageImages.delete(next);
        if(serial===coverageSerial){coverageKey=null;element('coverage-status').textContent='Dekningsgrad kunne ikke tegnes';}
      });
    }catch(failure){
      if(failure.name!=='AbortError'&&serial===coverageSerial){
        coverageKey=null;element('coverage-status').textContent=failure.message;
      }
    }
  },150);
}
async function inspectCoverage(location){
  if(busy)return true;
  const serial=++inspectionSerial,content=document.createElement('div');
  content.textContent='Laster dekningsgrad…';
  const popup=L.popup().setLatLng(location).setContent(content).openOn(map);
  try{
    const response=await fetch('/api/coverage/point?'+new URLSearchParams({lon:location.lng,lat:location.lat,
      classes:JSON.stringify(coverageClasses())}));
    const result=await response.json();
    if(!response.ok)throw new Error(result.error||'Dekningsgrad kunne ikke lastes');
    if(serial!==inspectionSerial||!map.hasLayer(popup))return true;
    if(!result.found){map.closePopup(popup);return false;}
    content.textContent=result.label;
  }catch(failure){if(serial===inspectionSerial)content.textContent=failure.message;}
  return true;
}
async function initializeCoverage(){
  try{
    const response=await fetch('/api/coverage/options');
    if(!response.ok)throw new Error('Dekningsgrad kunne ikke lastes');
    coverageOptions=await response.json();
    if(!coverageOptions.available){element('coverage-status').textContent='Ingen lagret dekningsgrad tilgjengelig.';return;}
    const previous=coverageClasses();
    const controls=element('coverage-classes');controls.replaceChildren();
    for(const item of coverageOptions.classes){
      const label=document.createElement('label'),input=document.createElement('input'),swatch=document.createElement('span');
      input.type='checkbox';input.value=item.value;input.checked=previous.length?previous.includes(item.value):true;
      input.onchange=scheduleCoverage;swatch.className='sensor-dot';swatch.style.background=item.color;
      label.append(input,swatch,document.createTextNode(item.label));
      controls.append(label);
    }
    element('coverage-show').disabled=false;scheduleCoverage();
  }catch(failure){element('coverage-status').textContent=failure.message;}
}
const element = id => document.getElementById(id);
const icons = () => { if (window.lucide) lucide.createIcons(); };
const number = (value, digits=4) => new Intl.NumberFormat('nb-NO',{minimumFractionDigits:digits,maximumFractionDigits:digits}).format(value);
const shortNumber = value => new Intl.NumberFormat('nb-NO',{maximumSignificantDigits:3}).format(value);
const overlayOpacity = () => 1-Number(element('opacity').value)/100;
const selectedNetwork = () => state.network==='frost'&&state.pross_only?(state.include_met?'svv_met':'svv'):state.network;
const selectedPlacementNetwork = () => ({frost:'all',svv:'svv',svv_met:'svv-met',empty:'empty'})[selectedNetwork()]||'greedy';
const terrainLabels={elevation:'Høyde over havet',coast_dist_fjord:'Avstand til fjord',coast_dist_ocean:'Avstand til åpent hav'};
const loadingTasks=new Map();
let loadingSerial=0,loadingStarted=0,loadingFinished=0,loadingFailed=false;
function beginLoading(label){
  if(!loadingTasks.size){loadingStarted=0;loadingFinished=0;loadingFailed=false;}
  const token=++loadingSerial;
  loadingTasks.set(token,label);loadingStarted++;renderLoading();return token;
}
function finishLoading(token,failed=false){
  if(!loadingTasks.delete(token))return;
  loadingFinished++;loadingFailed=loadingFailed||failed;renderLoading();
}
function renderLoading(){
  const active=loadingTasks.size>0,progress=element('load-progress');
  element('load-status').setAttribute('aria-busy',String(active));
  element('load-label').textContent=active?`${loadingTasks.values().next().value} · ${loadingFinished} av ${loadingStarted} forespørsler ferdige`:loadingFailed?'Lastingen er ferdig med feil':'Klar';
  progress.hidden=!active;
  if(loadingFinished){progress.max=loadingStarted;progress.value=loadingFinished;}
  else progress.removeAttribute('value');
}
const nativeFetch=window.fetch.bind(window);
const fetch=async(resource,options)=>{
  const url=String(resource);
  if(url.includes('/api/precompute'))return nativeFetch(resource,options);
  const label=url.includes('/api/snapshot-data')?'Laster ned værnettet':
    url.includes('/roads/')?'Laster vegnett':url.includes('/api/action')?'Oppdaterer scenario':url.includes('/api/state')?'Laster stasjoner og værdata':'Laster kartdata';
  const token=beginLoading(label);
  try{
    const response=await nativeFetch(resource,options);
    finishLoading(token,!response.ok);return response;
  }catch(problem){finishLoading(token,problem.name!=='AbortError');throw problem;}
};
icons();
element('save-map').onclick=saveFrozenMap;
element('save-scenario').onclick=saveEditableScenario;
element('load-scenario').onclick=()=>element('scenario-file').click();
element('scenario-file').onchange=event=>loadEditableScenario(event.target.files[0]);
let state = null, busy = false, mode = 'uncertainty', overlay = null, overlaySerial = 0;
let overlayKey=null,roadInitialization=null;
let progressBusy=false;
let committedLegendTitle='';
let roadOptions=null,roadResult=null,roadSerial=0,roadTimer=null,roadController=null;
let roadBusy=false,roadPending=false,roadRequestKey=null;
let stationsVisible=true;
let kuldehullAvailable=false,kuldehullOverlay=null,kuldehullSerial=0,kuldehullTimer=null;
let kuldehullRequestKey=null,kuldehullController=null,kuldehullUrl=null;
let inspectionSerial=0;
const terrainCatalogs=new Map(),terrainPending=new Set();
let climateCatalog=null;
const placementCatalogs=new Map(),placementPending=new Set();
const map = L.map('map', {zoomControl:false,minZoom:3,maxZoom:15}).setView([65,15],4);
L.control.zoom({position:'topright',zoomInTitle:'Zoom inn',zoomOutTitle:'Zoom ut'}).addTo(map);
const backgroundTiles=L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
  attribution:'&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',maxZoom:19
});
let tileLoadingToken=null;
backgroundTiles.on('loading',()=>{if(tileLoadingToken===null)tileLoadingToken=beginLoading('Laster bakgrunnskart');});
backgroundTiles.on('load',()=>{finishLoading(tileLoadingToken);tileLoadingToken=null;});
backgroundTiles.addTo(map);
L.control.scale({imperial:false,position:'bottomright'}).addTo(map);
const fixedMarkers = L.layerGroup().addTo(map), manualMarkers = L.layerGroup().addTo(map);
const calculationMarkers=L.layerGroup().addTo(map);
const inspectionMarkers=L.layerGroup().addTo(map);
map.on('popupclose',()=>inspectionMarkers.clearLayers());
map.createPane('roads');map.getPane('roads').style.zIndex=430;
map.createPane('stations');map.getPane('stations').style.zIndex=450;
map.createPane('calculation-points');map.getPane('calculation-points').style.zIndex=445;
map.createPane('county-borders');map.getPane('county-borders').style.zIndex=440;
map.getPane('county-borders').style.pointerEvents='none';
const countyBorders=L.geoJSON(null,{pane:'county-borders',interactive:false,
  style:{color:'#4b5359',weight:1,opacity:.8,fill:false}}).addTo(map);
map.createPane('background-roads');map.getPane('background-roads').style.zIndex=420;
map.getPane('background-roads').style.pointerEvents='none';
const backgroundRoads=L.tileLayer('/api/roads/background/{z}/{x}/{y}.png',{
  pane:'background-roads',maxZoom:22,maxNativeZoom:18,updateWhenIdle:true,
  keepBuffer:2,bounds:[[57,3],[72,33]]
});
let backgroundRoadKey=null;
let backgroundRoadError=false;
backgroundRoads.on('tileerror',()=>{
  if(!backgroundRoadError){backgroundRoadError=true;error('Bakgrunnsvegnettet kunne ikke lastes.');}
});
const roadRenderer=L.canvas({pane:'roads',padding:0.3});
const roadCasing=L.geoJSON(null,{pane:'roads',renderer:roadRenderer,interactive:false,style:{color:'#fff',weight:4,opacity:.85}}).addTo(map);
const roadLines=L.geoJSON(null,{pane:'roads',renderer:roadRenderer,bubblingMouseEvents:false,style:roadStyle,onEachFeature:(feature,layer)=>layer.on('click',async event=>{
  L.DomEvent.stopPropagation(event.originalEvent);clearPreview();
  if(event.originalEvent.ctrlKey||event.originalEvent.metaKey)inspectKuldehull(event.latlng);
  else if(event.originalEvent.altKey)inspectBestStation(event.latlng);
  else if(element('coverage-show').checked&&await inspectCoverage(event.latlng))return;
  else showRoadPopup(feature,event.latlng);
})}).addTo(map);

function error(message) {
  element('message').querySelector('span').textContent = message;
  element('message').hidden = false;
}
element('message').querySelector('button').onclick = () => element('message').hidden = true;
fetch('/api/counties/borders').then(async response=>{
  const result=await response.json();
  if(!response.ok)throw new Error(result.error||'Fylkesgrensene kunne ikke lastes');
  countyBorders.addData(result);
}).catch(problem=>error(problem.message));

function controls() {
  for(const id of ['save-map','save-scenario','load-scenario'])element(id).disabled=busy||!state;
  element('reset-defaults').disabled=busy||!state;
  document.body.classList.toggle('busy',busy);
  element('map').classList.toggle('busy',busy);
  element('network').disabled = busy || !state;
  element('phenomena').disabled = busy || !state;
  element('network').querySelector('[value="svv"]').disabled=!state?.pross_available;
  element('network').querySelector('[value="svv_met"]').disabled=!state?.met_available;
  element('station-since').disabled=busy||!state||state.network!=='frost';
  element('station-date-apply').disabled=busy||!stationDatePreview;
  document.querySelectorAll('#county-options input,#county-all,#county-none').forEach(input=>input.disabled=busy||!state);
  element('terrain-layer').disabled=busy||!state||!terrainCatalogs.has(state.cell_m);
  for(const id of ['kuldehull-show','kuldehull-only','kuldehull-depth'])element(id).disabled=busy||!state||!kuldehullAvailable;
  for (const id of ['undo','reset']) element(id).disabled = busy || !state?.stations.length;
  document.querySelectorAll('[data-remove]').forEach(button => button.disabled = busy);
  element('comparison-baseline').disabled=busy||!state;
  element('export').setAttribute('aria-disabled',String(busy || !state || state.mean_rmse===null));
}

function clearPreview() {
  if (overlay) overlay.setOpacity(overlayOpacity());
  if(committedLegendTitle)element('legend-title').textContent=committedLegendTitle;
}

async function requestState(action) {
  if (busy) return;
  cancelStationDate();
  clearPreview(); busy=true; controls(); element('message').hidden=true;
  try {
    const response=await fetch(action?'/api/action':'/api/state',action?{
      method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(action)
    }:{});
    const result=await response.json();
    if (!response.ok) throw new Error(result.error || 'Scenariet kunne ikke oppdateres');
    if(action?.action==='add'&&mode==='baseline')mode='uncertainty';
    state=result; render();return true;
  } catch(failure) {error(failure.message);if(state)render();return false;}
  finally {busy=false;controls();}
}

function render() {
  if(state.colour_reference_warning)error(state.colour_reference_warning);
    map.closePopup();
    element('comparison-tabs').hidden=state.network!=='frost';
    element('comparison-baseline').setAttribute('aria-pressed',String(state.comparison_scope==='baseline'));
    const [phenomenon,kind]=state.dataset.split(':');
    const options=new Map((state.layer_options||[]).map(item=>[item.phenomenon,item]));
    element('phenomena').querySelectorAll('input[data-phenomenon]').forEach(input=>{
      const option=options.get(input.value);
      input.checked=state.layers.length>0&&(Boolean(option&&state.layers.includes(option.dataset))||(!option&&input.value===phenomenon));
      input.disabled=!option?.available;
      input.closest('label').classList.toggle('unavailable',!option?.available);
      input.closest('label').title=option?.available?'':'Ingen ferdig modell';
    });
    element('layers-note').hidden=state.layers.length<2&&!state.missing_layers?.length;
      (state.missing_layers||[]).map(layer=>` Mangler lagret grunnflate for ${layer.label} på ${number(state.cell_m,0)} m${layer.error?` · ${layer.error}`:''}.`).join('');
    element('station-since').value=state.station_since||'';
    element('station-since').disabled=busy||state.network!=='frost';
    renderCountySelection();
    const sensorKey=element('sensor-key');
    if(sensorKey){
    sensorKey.replaceChildren();
    for(const option of (state.layer_options||[]).filter(item=>item.available)){
      const item=document.createElement('span');item.append(sensorDot(option.phenomenon),SENSORS[option.phenomenon]?.[0]||option.phenomenon);sensorKey.append(item);
    }
    if(state.pross_available){
      const item=document.createElement('span'),box=document.createElement('span');box.className='svv-key';
      item.append(box,'vegværstasjon');sensorKey.append(item);
    }
    }
    element('network').value=selectedNetwork();
    element('station-radius-note').hidden=!state.station_radius_m;
    //element('station-radius-note').textContent=state.station_radius_m?
    //  `Eksperiment: nye/flyttede stasjoner påvirker bare ruter innen ${number(state.station_radius_m/1000,0)} km. Grunnnettet er uendret.`:'';
    element('network-count').textContent=`${number(state.fixed_count+state.stations.length,0)} målepunkter`;
    element('manual-count').textContent=state.stations.length;
    document.title=`Værnett`;
    showProgress(state.precompute_jobs||[]);
    element('class-definition').hidden=kind!=='class';
    const selected=state.selected_station;
    fixedMarkers.clearLayers();manualMarkers.clearLayers();
    const checkedSensors=new Set(state.layers.map(dataset=>dataset.split(':')[0]));
    const hasCheckedSensor=station=>checkedSensors.size>0&&(!station.sensors||!state.layer_options?.length||station.sensors.some(sensor=>checkedSensors.has(sensor)));
    for(const station of state.fixed_stations){
      const proposal=sensorProposal(station);
      const allSensors=[...new Set([...(station.sensors||[]),...(proposal?.sensors||[])])];
      const inactive=!hasCheckedSensor({...station,sensors:allSensors});
      const tooltip=document.createElement('span');
      tooltip.textContent=station.kind==='observation'?`${station.station_id} · ${station.name}`:`Foreslått stasjon ${station.rank}`;
      //tooltip.textContent=station.kind==='observation'?`${station.station_id} · ${station.name} · ${station.first_time.slice(0,10)}–${station.last_time.slice(0,10)}`:`Foreslått stasjon ${station.rank}`;
      if(station.removed)tooltip.textContent+=' · Fjernet fra scenariet';
      const selectionKind=station.kind==='observation'?'frost':'greedy';
      const selectionIndex=station.kind==='observation'?station.index:station.rank-1;
      const isSelected=(proposal&&selected?.kind==='manual'&&selected.index===proposal.id)||(station.kind==='observation'?selected?.kind==='frost'&&selected.station_id===station.station_id:selected?.kind===selectionKind&&selected.index===selectionIndex);
      if(station.kind==='observation'){
        if(station.original_position){
          const original=L.latLng(station.original_position.lat,station.original_position.lon);
          const current=L.latLng(station.lat,station.lon);
          const start=map.project(original),end=map.project(current);
          const midpoint=map.unproject(start.add(end).divideBy(2));
          const angle=Math.atan2(end.y-start.y,end.x-start.x)*180/Math.PI;
          L.polyline([original,current],{pane:'stations',color:'#697470',weight:2,dashArray:'6 5',interactive:false}).addTo(fixedMarkers);
          const arrow=L.divIcon({className:'station-move-arrow',html:`<svg width="20" height="20" viewBox="0 0 20 20" style="transform:rotate(${angle}deg)" aria-hidden="true"><path d="M3 3 L17 10 L3 17 L7 10 Z" fill="#697470" stroke="#fff" stroke-width="1.5"/></svg>`,iconSize:[20,20],iconAnchor:[10,10]});
          L.marker(midpoint,{pane:'stations',icon:arrow,interactive:false,keyboard:false}).addTo(fixedMarkers);
          const originalIcon=L.divIcon({className:`frost-pin original${station.svv?' svv':''}${isSelected?' selected':''}`,html:'',iconSize:[14,14],iconAnchor:[7,7]});
          const originalLabel=`${station.station_id} · ${station.name} · Opprinnelig plassering`;
          const originalTooltip=document.createElement('span');originalTooltip.textContent=originalLabel;
          L.marker(original,{pane:'stations',icon:originalIcon,savedStation:station,originalStationPosition:true,bubblingMouseEvents:false,title:originalLabel,zIndexOffset:isSelected?900:400})
            .bindTooltip(originalTooltip).on('click',event=>{L.DomEvent.stopPropagation(event.originalEvent);chooseFrostStation(station,event.latlng);}).addTo(fixedMarkers);
          tooltip.textContent+=' · Flyttet plassering';
        }
        const draggable=element('station-move-mode').checked&&state.frost_ready&&!station.removed&&station.editable;
        const pie=station.removed?'':sensorPie(allSensors);
        const icon=L.divIcon({className:`frost-pin${proposal||station.original_position||station.sensors_changed?' altered-station':''}${inactive?' inactive-sensors':''}${station.svv?' svv':''}${isSelected?' selected':''}${station.removed?' removed':''}${draggable?'':' locked'}`,html:pie?`<span class="sensor-pie" style="background:${pie}"></span>`:'',iconSize:[14,14],iconAnchor:[7,7]});
        const marker=L.marker([station.lat,station.lon],{pane:'stations',icon,draggable,savedStation:station,zIndexOffset:isSelected?1000:station.removed?500:0,bubblingMouseEvents:false,title:draggable?'Dra for å flytte værstasjonen':'Skru på flyttemodus for å flytte stasjonen'}).bindTooltip(tooltip).addTo(fixedMarkers);
        marker.on('click',event=>{L.DomEvent.stopPropagation(event.originalEvent);chooseFrostStation(station,event.latlng);});
        marker.on('dragstart',clearPreview);
        marker.on('dragend',event=>{const point=event.target.getLatLng();requestState(
          {action:'move_frost',station_id:station.station_id,lon:point.lng,lat:point.lat});});
      }else{
        const marker=L.circleMarker([station.lat,station.lon],{pane:'stations',radius:isSelected?6:3,color:isSelected?'#edb138':'#fff',weight:isSelected?2:.8,fillColor:'#35463f',fillOpacity:.95,bubblingMouseEvents:false}).bindTooltip(tooltip).addTo(fixedMarkers);
        marker.on('click',event=>{L.DomEvent.stopPropagation(event.originalEvent);selectStation(selectionKind,selectionIndex);});
      }
    }
    element('stations').replaceChildren();
    state.stations.forEach((station,index)=>{
      const attached=state.fixed_stations.some(original=>sensorProposal(original)?.id===station.id);
      const isSelected=selected?.kind==='manual'&&selected.index===station.id;
      const manualPie=sensorPie(station.sensors||[]);
      const icon=L.divIcon({className:`new-pin${!hasCheckedSensor(station)?' inactive-sensors':''}${isSelected?' selected':''}`,html:`<span class="new-pin-pie"${manualPie?` style="background:${manualPie}"`:''}>${index+1}</span>`,iconSize:[24,24],iconAnchor:[12,12]});
      if(!attached)L.marker([station.lat,station.lon],{icon,draggable:element('station-move-mode').checked,zIndexOffset:1000,title:element('station-move-mode').checked?'Dra for å flytte værstasjonen':'Skru på flyttemodus for å flytte stasjonen',alt:`Nytt målepunkt ${index+1}`,bubblingMouseEvents:false}).bindTooltip(`Nytt målepunkt ${index+1}`).on('click',async event=>{
        L.DomEvent.stopPropagation(event.originalEvent);
        await openManualStation(station,index);
      }).on('dragstart',clearPreview).on('dragend',async event=>{
        const point=event.target.getLatLng();
        await requestState({action:'move_manual',id:station.id,lon:point.lng,lat:point.lat});
      }).addTo(manualMarkers);
      const item=document.createElement('li');item.className='station';
      const badge=document.createElement('span');badge.className='station-number';badge.textContent=index+1;
      const info=document.createElement('button');info.className='station-select';info.setAttribute('aria-pressed',String(isSelected));info.title=`Vis ny stasjon ${index+1} og endre sensorer`;info.onclick=()=>openManualStation(station,index);
      const name=document.createElement('strong');name.textContent=`Nytt målepunkt ${index+1}`;
      const coordinates=document.createElement('small');coordinates.textContent=`${number(station.lat)}° N  ${number(station.lon)}° Ø`;
      const sensorLine=document.createElement('small');sensorLine.className='station-sensors';
      for(const sensor of station.sensors||[])sensorLine.append(sensorDot(sensor));
      sensorLine.title=(station.sensors||[]).map(sensor=>SENSORS[sensor]?.[0]||sensor).join(', ');
      info.append(name,coordinates,sensorLine);
      const remove=document.createElement('button');remove.className='icon-button';remove.dataset.remove=station.id;remove.title=`Fjern nytt målepunkt ${index+1}`;remove.setAttribute('aria-label',remove.title);remove.innerHTML='<i data-lucide="trash-2"></i>';remove.onclick=()=>requestState({action:'remove',id:station.id});
      item.append(badge,info,remove);element('stations').append(item);
    });
    //element('empty').hidden=state.stations.length>0;
    renderCalculationPoints();
    icons();loadTerrainOptions();updateMap();scheduleRoads();scheduleKuldehull();scheduleCoverage();
  }

function renderCalculationPoints(){
  calculationMarkers.clearLayers();
  if(!state||!stationsVisible||!element('station-calculation-points').checked)return;
  buildCalculationPoints(calculationMarkers);
}
function buildCalculationPoints(group,includeAll=false){
  const sensors=new Set(includeAll?Object.keys(SENSORS):state.layers.map(dataset=>dataset.split(':')[0]));
  for(const station of [...state.fixed_stations,...state.stations]){
    if(station.removed||!sensors.size||station.sensors&&!station.sensors.some(sensor=>sensors.has(sensor)))continue;
    const seen=new Set();
    for(const point of station.calculation_positions||[]){
      if(!includeAll&&point.dataset&&!state.layers.includes(point.dataset))continue;
      const key=`${point.lon},${point.lat}`;
      if(seen.has(key))continue;
      seen.add(key);
      L.polyline([[station.lat,station.lon],[point.lat,point.lon]],{
        pane:'calculation-points',color:'#596861',weight:1,opacity:.65,dashArray:'2 4',interactive:false
      }).addTo(group);
      const label=document.createElement('span');
      label.textContent=`${station.station_id||'Nytt målepunkt'} · Beregningspunkt · ${number(point.cell_m,0)} m-rute`;
      L.circleMarker([point.lat,point.lon],{pane:'calculation-points',radius:3,color:'#596861',
        weight:1,fillColor:'#fff',fillOpacity:.75,bubblingMouseEvents:false
      }).bindTooltip(label).addTo(group);
    }
  }
}

let activeMapImage=null,pendingMapImage=null;
let placementTileLayer=null,placementTileKey=null,placementTileLoading=null,placementPaintFrame=null;
let uncertaintyTileLayer=null,uncertaintyTileKey=null,uncertaintyTileLoading=null;
function paintRasterCanvas(canvas,cutoff){
  if(!canvas.rasterValues||canvas.rasterCutoff===cutoff)return;
  const pixels=canvas.rasterPixels.data,source=canvas.rasterColours,values=canvas.rasterValues;
  for(let i=0;i<values.length;i++)pixels[i*4+3]=cutoff==null||values[i]>cutoff?source[i*4+3]:0;
  canvas.getContext('2d').putImageData(canvas.rasterPixels,0,0);
  canvas.rasterCutoff=cutoff;
}
async function loadRasterCanvas(canvas,url,signal,cutoff=undefined){
  const response=await nativeFetch(url,{signal});
  if(!response.ok)throw new Error('Kartdataene kunne ikke lastes');
  const buffer=await response.arrayBuffer();
  if(buffer.byteLength<8)throw new Error('Kartlaget har ugyldige data');
  const header=new DataView(buffer),width=header.getUint32(0,true),height=header.getUint32(4,true),count=width*height;
  if(!width||!height||buffer.byteLength!==8+count*8)throw new Error('Kartlaget har ugyldige data');
  canvas.width=width;canvas.height=height;
  canvas.rasterValues=new Float32Array(buffer,8,count);
  canvas.rasterColours=new Uint8ClampedArray(buffer,8+count*4,count*4);
  canvas.rasterPixels=canvas.getContext('2d').createImageData(width,height);
  canvas.rasterPixels.data.set(canvas.rasterColours);
  paintRasterCanvas(canvas,cutoff===undefined?(rmseMinimum()||null):cutoff);
}
function repaintMapRaster(){
  if(placementPaintFrame!==null)return;
  placementPaintFrame=requestAnimationFrame(()=>{
    placementPaintFrame=null;
    const cutoff=rmseMinimum()||null;
    const tileLayers=[placementTileLayer,uncertaintyTileLayer].filter(Boolean);
    if(tileLayers.length){
      for(const layer of tileLayers)for(const tile of Object.values(layer._tiles))paintRasterCanvas(tile.el,cutoff);
    }else if(overlay?.rasterCanvas)paintRasterCanvas(overlay.rasterCanvas,cutoff);
  });
}
function rasterCanvasOverlay(canvas,bounds,options){
  const Raster=L.ImageOverlay.extend({
    _initImage(){
      this._image=canvas;
      L.DomUtil.addClass(canvas,`leaflet-image-layer ${this._zoomAnimated?'leaflet-zoom-animated':''}`);
      canvas.onselectstart=L.Util.falseFn;canvas.onmousemove=L.Util.falseFn;
    }
  });
  const layer=new Raster('',bounds,options);
  layer.rasterCanvas=canvas;
  return layer;
}
function createRasterOverlay(url,bounds,options){
  const canvas=document.createElement('canvas'),controller=new AbortController();
  const layer=rasterCanvasOverlay(canvas,bounds,options);
  layer.loadRaster=()=>loadRasterCanvas(canvas,url,controller.signal).then(()=>layer.fire('load'))
    .catch(failure=>{if(failure.name!=='AbortError')layer.fire('error',{error:failure});});
  layer.once('remove',()=>controller.abort());
  return layer;
}
function createRasterTileLayer(url,options){
  const Tiles=L.GridLayer.extend({
    createTile(coords,done){
      const canvas=document.createElement('canvas');canvas.width=canvas.height=256;
      const controller=new AbortController();canvas.placementController=controller;
      loadRasterCanvas(canvas,L.Util.template(url,coords),controller.signal).then(()=>done(null,canvas))
        .catch(failure=>{if(failure.name!=='AbortError')done(failure,canvas);});
      return canvas;
    }
  });
  const layer=new Tiles(options);
  layer.on('tileunload',event=>event.tile.placementController?.abort());
  return layer;
}
function removePlacementTiles(){
  if(!placementTileLayer)return;
  map.removeLayer(placementTileLayer);
  if(overlay===placementTileLayer)overlay=null;
  placementTileLayer=null;placementTileKey=null;
  if(placementPaintFrame!==null){cancelAnimationFrame(placementPaintFrame);placementPaintFrame=null;}
  if(placementTileLoading!==null){finishLoading(placementTileLoading);placementTileLoading=null;}
}
function removeUncertaintyTiles(){
  if(!uncertaintyTileLayer)return;
  map.removeLayer(uncertaintyTileLayer);
  if(overlay===uncertaintyTileLayer)overlay=null;
  uncertaintyTileLayer=null;uncertaintyTileKey=null;
  if(uncertaintyTileLoading!==null){finishLoading(uncertaintyTileLoading);uncertaintyTileLoading=null;}
}
function showPlacementTiles(layer,settings){
  const roads=settings?Object.fromEntries(Object.entries(settings).filter(([key])=>key!=='bounds')):null;
  const query=new URLSearchParams({dataset:state.dataset,network:selectedPlacementNetwork(),version:layer.tile_version,
    counties:JSON.stringify(state.counties)});
  if(roads)query.set('roads',JSON.stringify(roads));
  const key=query.toString();
  if(placementTileKey===key&&placementTileLayer){repaintMapRaster();return;}
  overlaySerial++;overlayKey=null;cancelQueuedMapImage();removePlacementTiles();
  if(overlay){map.removeLayer(overlay);overlay=null;}
  placementTileKey=key;
  const next=createRasterTileLayer(`/api/placement/tiles/{z}/{x}/{y}.bin?${key}`,{
    pane:'overlayPane',opacity:overlayOpacity(),bounds:layer.bounds,noWrap:true,maxNativeZoom:12,maxZoom:18,
    updateWhenIdle:true,keepBuffer:2
  });
  let failed=false;
  next.on('loading',()=>{
    if(next!==placementTileLayer)return;
    failed=false;
    if(placementTileLoading===null)placementTileLoading=beginLoading('Laster plasseringsgevinst');
  });
  next.on('tileerror',()=>{
    if(next!==placementTileLayer||failed)return;
    failed=true;
    error('Plasseringsgevinst kunne ikke tegnes. Last siden på nytt hvis kartdataene er endret.');
  });
  next.on('load',()=>{
    if(next!==placementTileLayer)return;
    if(placementTileLoading!==null){finishLoading(placementTileLoading,failed);placementTileLoading=null;}
  });
  placementTileLayer=next;overlay=next;next.addTo(map);
}
function showUncertaintyTiles(layerMode,settings){
  const roads=settings?Object.fromEntries(Object.entries(settings).filter(([key])=>key!=='bounds')):null;
  const query=new URLSearchParams({revision:String(state.revision),mode:layerMode,dataset:state.dataset});
  if(roads)query.set('roads',JSON.stringify(roads));
  const key=query.toString();
  if(uncertaintyTileKey===key&&uncertaintyTileLayer){repaintMapRaster();return;}
  overlaySerial++;overlayKey=null;cancelQueuedMapImage();removeUncertaintyTiles();
  if(overlay){map.removeLayer(overlay);overlay=null;}
  uncertaintyTileKey=key;
  const next=createRasterTileLayer(`/api/map/tiles/{z}/{x}/{y}.bin?${key}`,{
    pane:'overlayPane',opacity:overlayOpacity(),bounds:state.bounds,noWrap:true,maxNativeZoom:12,maxZoom:18,
    updateWhenIdle:true,keepBuffer:2
  });
  let failed=false;
  next.on('loading',()=>{
    if(next!==uncertaintyTileLayer)return;
    failed=false;
    if(uncertaintyTileLoading===null)uncertaintyTileLoading=beginLoading(
      layerMode==='reduction'?'Laster ruter for endring i usikkerhet':'Laster usikkerhetsruter');
  });
  next.on('tileerror',()=>{
    if(next!==uncertaintyTileLayer||failed)return;
    failed=true;
    error('Kartflisene kunne ikke tegnes. Last siden på nytt hvis scenariet er endret.');
  });
  next.on('load',()=>{
    if(next!==uncertaintyTileLayer)return;
    if(uncertaintyTileLoading!==null){finishLoading(uncertaintyTileLoading,failed);uncertaintyTileLoading=null;}
  });
  uncertaintyTileLayer=next;overlay=next;next.addTo(map);
}
function cancelQueuedMapImage(){pendingMapImage=null;}
function queueMapImage(url,bounds,serial,label,failureMessage,numeric=false){
  pendingMapImage={url,bounds,serial,label,failureMessage,numeric};
  if(activeMapImage)return;
  startQueuedMapImage();
}
function startQueuedMapImage(){
  if(!pendingMapImage)return;
  const job=pendingMapImage;pendingMapImage=null;
  const token=beginLoading(job.label);
  const next=(job.numeric?createRasterOverlay(job.url,job.bounds,{opacity:0,interactive:false}):
    L.imageOverlay(job.url,job.bounds,{opacity:0,interactive:false})).addTo(map);
  activeMapImage=next;
  function finish(failed){
    const current=job.serial===overlaySerial;
    finishLoading(token,failed&&current);
    if(failed||!current){
      map.removeLayer(next);
      if(failed&&current){overlayKey=null;error(job.failureMessage);}
    }else{
      if(overlay)map.removeLayer(overlay);
      overlay=next;overlay.setOpacity(overlayOpacity());
    }
    activeMapImage=null;
    startQueuedMapImage();
  }
  next.once('load',()=>finish(false));
  next.once('error',()=>finish(true));
  if(job.numeric)next.loadRaster();
}

function usesFixedP95Scale(){
  return state?.dataset==='temperature:direct_p95'&&state.layers.length===1&&
    ['uncertainty','baseline'].includes(mode);
}
function usesEmptyReferenceLegend(){
  return mode==='reduction'&&state?.network==='empty'&&['RMSE','P95'].includes(state.metric)&&
    state.dataset.startsWith('temperature:')&&state.layers.length===1;
}

function updateMap() {
  if(!state)return;
  const tileMode=['uncertainty','baseline','reduction'].includes(mode);
  if(mode!=='placement'||state.layers.length!==1)removePlacementTiles();
  if(!tileMode)removeUncertaintyTiles();
  const emptyReference=usesEmptyReferenceLegend();
  const emptyP95=emptyReference&&state.metric==='P95';
  clearPreview();
  const climate=mode==='climate';
  renderRoadSummary();
  element('climate-legend').hidden=!climate;
  element('ramp').hidden=climate;
  element('legend-numbers').hidden=true;
  element('low').parentElement.hidden=climate||emptyP95;
  element('low').parentElement.style.display=climate||emptyP95?'none':'';
  if(!state.layers.length&&!climate&&!mode.startsWith('terrain:')){
    overlaySerial++;overlayKey=null;cancelQueuedMapImage();
    if(overlay)map.removeLayer(overlay);
    overlay=null;
    element('legend-title').textContent='Ingen værfenomen valgt';
    element('ramp').hidden=true;element('low').parentElement.style.display='none';
    element('rmse-filter').hidden=true;document.title='Værnett';
    return;
  }
  element('rmse-filter').hidden=(!['RMSE','P95'].includes(state.metric)&&state.layers.length<2)||climate;
  refreshRmseRange();
  renderUncertaintyCutoff();
  document.querySelectorAll('[data-mode]').forEach(button=>button.setAttribute('aria-pressed',String(button.dataset.mode===mode)));
  if(climate){updateClimateMap();return;}
  const stationLabel=state.selected_station_label||'Ingen stasjon valgt';
  const weatherTitle=usesFixedP95Scale()?DIRECT_P95_LEGEND_TITLE:state.weather_label;
  const titles={uncertainty:weatherTitle,baseline:weatherTitle,reduction:`Redusert usikkerhet · ${stationLabel}`,affected:stationLabel};
  const baselineChange=state.comparison!=='contribution'&&['reduction','affected'].includes(mode);
  const comparisonTitle={baseline:'Endring fra opprinnelig grunnnett',move:`${stationLabel} · effekt av flytting`,removal:`${stationLabel} · økt usikkerhet ved fjerning`};
  if(baselineChange)titles[mode]=comparisonTitle[state.comparison];
  element('legend-title').textContent=titles[mode];
  const uncertainty=mode==='uncertainty'||mode==='baseline';
  if(uncertainty&&state.network==='frost')element('legend-title');
  const terrain=mode.startsWith('terrain:');
  const placement=mode==='placement';
  if(emptyReference){
    element('legend-title').textContent=emptyReductionLegendTitle(state.empty_reference_rmse,state.metric==='P95');
  }
  const placementLayer=placement?placementCatalogs.get(JSON.stringify([state.dataset,selectedPlacementNetwork()])):null;
  const terrainLayer=terrain?terrainCatalogs.get(state.cell_m)?.layers.find(layer=>layer.field===mode.split(':')[1]):null;
  element('terrain-layer').value=placement?'placement':terrain?mode.split(':')[1]:'';
  element('middle').hidden=!terrainLayer?.field.startsWith('tpi_');
  if(baselineChange){element('middle').hidden=false;element('middle').textContent='0';}
  if(terrain&&!terrainLayer){
    overlaySerial++;overlayKey=null;cancelQueuedMapImage();overlay?.setOpacity(0);element('legend-title').textContent='Laster terrenglag…';loadTerrainOptions();return;
  }
  if(placement&&!placementLayer){
    overlaySerial++;overlayKey=null;cancelQueuedMapImage();removePlacementTiles();overlay?.setOpacity(0);element('legend-title').textContent='Laster lagret plasseringsgevinst…';loadPlacementOptions();return;
  }
  element('ramp').style.background=uncertainty?UNCERTAINTY_LEGEND.gradient:mode==='affected'?'#d9007a':'linear-gradient(to right,#ffffd9,#7fcdbb,#225ea8,#081d58)';
  element('low').textContent=UNCERTAINTY_LEGEND.low;
  element('high').textContent=UNCERTAINTY_LEGEND.high;
  if(!uncertainty){element('low').textContent='Mindre forbedring\u00A0\u00A0\u00A0';element('high').textContent='Større forbedring';}
  if(emptyReference){element('low').textContent='';element('high').textContent='';}
  if(baselineChange){
    element('ramp').style.background='linear-gradient(to right,#053061,#6bacd1,#f7f7f7,#e48065,#67001f)';
    element('low').textContent='Mindre usikkert';
    element('high').textContent='Mer usikkert';
    element('middle').hidden=true;
  }
  if(terrainLayer){
    element('legend-title').textContent=`${terrainLayer.label}`;
    element('ramp').style.background=terrainLayer.gradient;
    element('low').textContent=shortNumber(terrainLayer.minimum);element('high').textContent=shortNumber(terrainLayer.maximum);
    element('domain').textContent=`Norge / ${terrainLayer.label}`;document.title=`Værnett | ${terrainLayer.label}`;
  }else if(placementLayer){
    element('legend-title').textContent=`Plasseringsgevinst (°C·km²)`;
    element('ramp').style.background='linear-gradient(to right,#ffffcc,#feec9f,#fed875,#fdb14b,#fc8c3b,#fb4c29,#e2191c,#bb0026,#800026)';
    element('low').textContent='Mindre gevinst';element('high').textContent='Mer gevinst';
    element('domain').textContent=`Norge / ${placementLayer.weather_label} / Plasseringsgevinst`;
    document.title=`Værnett | ${placementLayer.weather_label} · Plasseringsgevinst`;
  }else{
    element('domain').textContent=`Norge / ${state.weather_label}`;document.title=`Værnett | ${state.weather_label}`;
  }
  const combined=!terrain&&state.layers.length>1;
  if(combined){
    const names=state.layer_labels.join(' + ');
    element('legend-title').textContent=names;
    element('ramp').style.background=uncertainty||baselineChange?'linear-gradient(to right,#053061,#6bacd1,#f7f7f7,#e48065,#67001f)':'linear-gradient(to right,#ffffd9,#7fcdbb,#225ea8,#081d58)';
    element('middle').hidden=true;
    element('domain').textContent=`Norge / ${names}`;document.title=`Værnett | ${names}`;
  }
  committedLegendTitle=element('legend-title').textContent;
  refreshLegendNumbers();
  const roadSettings=getWeatherRoadSettings();
  if(placement&&!combined){showPlacementTiles(placementLayer,roadSettings);return;}
  if(tileMode){showUncertaintyTiles(mode,roadSettings);return;}
  const corridorQuery=roadSettings?.corridor?`&roads=${encodeURIComponent(JSON.stringify(roadSettings))}`:'';
  const kuldehull=getKuldehullSettings();
  const kuldehullQuery=!placement&&!terrain&&kuldehull.only?`&kuldehull=${encodeURIComponent(JSON.stringify(kuldehull))}`:'';
  const size=map.getZoom()<=5?400:map.getZoom()<=7?800:1600;
  const savedFrost=!combined&&!state.station_since&&state.counties===null&&state.network==='frost'&&!state.frost_edits?.length&&['uncertainty','baseline'].includes(mode)&&(mode==='baseline'||!state.stations.length);
  const numeric=uncertainty;
  const rmseQuery=!numeric&&!terrain&&rmseMinimum()?`&rmse_min=${rmseMinimum()}`:'';
  const layersQuery=`&layers=${encodeURIComponent(state.layers.join(','))}`;
  const view=map.getBounds();
  const placementBounds=placement&&placementLayer.cell_m===500?
    [Math.max(-180,view.getWest()),Math.max(-85,view.getSouth()),Math.min(180,view.getEast()),Math.min(85,view.getNorth())]:null;
  const key=JSON.stringify([mode,state.dataset,state.layers,state.combine,state.counties,state.pross_only,state.include_met,placement?placementLayer.cell_m:state.cell_m,placement?'saved':state.network,placement&&!combined?placementLayer.source:savedFrost?state.frost_ready:state.revision,size,corridorQuery,kuldehullQuery,rmseQuery,placementBounds]);
  if(overlayKey===key&&(overlay||activeMapImage||pendingMapImage)){if(numeric)repaintMapRaster();return;}
  overlayKey=key;
  const serial=++overlaySerial;
  const placementQuery=placementBounds?`&bounds=${encodeURIComponent(JSON.stringify(placementBounds))}`:'';
  const imageUrl=placement?`/api/placement.png?dataset=${encodeURIComponent(state.dataset)}${layersQuery}&revision=${state.revision}&size=${size}${placementQuery}${corridorQuery}${rmseQuery}`:`/api/map.${numeric?'bin':'png'}?mode=${mode}&revision=${state.revision}&dataset=${encodeURIComponent(state.dataset)}${layersQuery}&cell_m=${state.cell_m}&size=${size}${corridorQuery}${kuldehullQuery}${rmseQuery}`;
  const imageBounds=kuldehullQuery?[[kuldehull.bounds[1],kuldehull.bounds[0]],[kuldehull.bounds[3],kuldehull.bounds[2]]]:placementBounds?[[placementBounds[1],placementBounds[0]],[placementBounds[3],placementBounds[2]]]:placement?placementLayer.bounds:state.bounds;
  queueMapImage(imageUrl,imageBounds,serial,'Laster værkart','Kartlaget kunne ikke tegnes. Stasjonsscenariet er uendret.',numeric);
}

let legendNumbersKey=null,legendNumbersSerial=0,legendNumbersReady=false;
async function refreshLegendNumbers(){
  const row=element('legend-numbers');
  if(!state||element('ramp').hidden){
    row.hidden=true;legendNumbersKey=null;legendNumbersSerial++;return;
  }
  const changeLegend=mode==='reduction';
  const placementLegend=mode==='placement';
  const signedChange=changeLegend&&state.comparison!=='contribution';
  const emptyReference=usesEmptyReferenceLegend();
  const emptyP95=emptyReference&&state.metric==='P95';
  const count=placementLegend?3:changeLegend?(signedChange?3:2):5;
  const key=JSON.stringify([count,mode,state.revision,state.network,state.comparison,state.dataset,state.layers,state.combine,state.cell_m,
    state.color_min,state.color_max,state.reduction_max]);
  if(key===legendNumbersKey){row.hidden=!legendNumbersReady;return;}
  legendNumbersKey=key;legendNumbersReady=false;row.hidden=true;
  const serial=++legendNumbersSerial,revision=state.revision,requestedMode=mode;
  try{
    const response=await fetch(`/api/legend?mode=${encodeURIComponent(mode)}&count=${count}`),result=await response.json();
    if(!response.ok)throw new Error(result.error||'Fargeskalaens verdier kunne ikke lastes');
    if(serial!==legendNumbersSerial||mode!==requestedMode||state.revision!==revision||
        element('ramp').hidden)return;
    if(result.revision!==revision)throw new Error('Fargeskalaen er endret. Oppdater kartet.');
    row.replaceChildren();
    const values=emptyP95?[state.empty_reference_rmse,0]:result.values;
    values.forEach((value,index)=>{
      const label=document.createElement('span');
      const prefix=emptyP95?(index===0?'≥':''):placementLegend?'':changeLegend?
        (index===0&&signedChange?'\u2264':index===result.values.length-1&&!emptyReference?'\u2265':''):
        (index===0&&!result.fixed?'\u2264':index===result.values.length-1?'\u2265':'');
      const formatted=placementLegend?number(value,0):Math.abs(value)>=10000||value!==0&&Math.abs(value)<.001?
        value.toExponential(1).replace('.',',').replace('e+','e'):shortNumber(value);
      const sign=signedChange&&value>0?'+':'';
      label.textContent=prefix+sign+formatted+(changeLegend&&!emptyP95?` ${result.unit}`:'');
      label.style.left=`${index/(values.length-1)*100}%`;row.append(label);
    });
    row.setAttribute('aria-label',`Fargeskala (${result.unit})`);
    legendNumbersReady=values.length>0;row.hidden=!legendNumbersReady;
  }catch(failure){
    if(serial===legendNumbersSerial){legendNumbersKey=null;error(failure.message);}
  }
}
function selectStation(kind,index){requestState({action:'select',kind,index});}
async function chooseFrostStation(station,latlng){
  const proposal=sensorProposal(station);
  const activeProposal=proposal?.sensors.some(sensor=>state.layers.some(dataset=>dataset.startsWith(`${sensor}:`)));
  await requestState(activeProposal?{action:'select',kind:'manual',index:proposal.id}:
    {action:'select',kind:'frost',station_id:station.station_id});
  const current=state?.fixed_stations.find(item=>item.station_id===station.station_id)||station;
  openStationPopup(latlng,current);
}
const SENSORS={temperature:['Temperatur','#d7301f'],wind:['Vind','#33a02c'],snow:['Snødybde','#6b4226'], precipitation:['Nedbør','#1f78b4'],humidity:['Luftfuktighet','#ff7f00'],road_temperature:['Vegbanetemperatur','#8c6bb1']};
function sensorProposal(station){
  if(station.kind!=='observation'||station.removed)return null;
  return state.stations.find(proposal=>Math.abs(proposal.lon-station.lon)<1e-9&&Math.abs(proposal.lat-station.lat)<1e-9&&
    proposal.sensors.some(sensor=>!(station.sensors||[]).includes(sensor)))||null;
}
function sensorPie(sensors){
  const colours=sensors.filter(sensor=>SENSORS[sensor]).map(sensor=>SENSORS[sensor][1]);
  if(!colours.length)return '';
  if(colours.length===1)return colours[0];
  const step=360/colours.length;
  return `conic-gradient(${colours.map((colour,index)=>`${colour} ${index*step}deg ${(index+1)*step}deg`).join(',')})`;
}
function sensorDot(sensor){
  const dot=document.createElement('span');dot.className='sensor-dot';dot.style.background=SENSORS[sensor]?.[1]||'#929995';dot.title=SENSORS[sensor]?.[0]||sensor;return dot;
}
async function openManualStation(station,index){
  if(!await requestState({action:'select',kind:'manual',index:station.id}))return;
  const current=state.stations.find(item=>item.id===station.id);
  if(!current)return;
  openStationPopup(L.latLng(current.lat,current.lon),{...current,kind:'manual',
    station_id:`Nytt målepunkt ${index+1}`,
    layers:(current.sensors||[]).filter(sensor=>state.layers.some(dataset=>dataset.startsWith(`${sensor}:`)))});
}
function openStationPopup(latlng,station,selectionKind,selectionIndex){
  const proposal=sensorProposal(station);
  const addedSensors=(proposal?.sensors||[]).filter(sensor=>!(station.sensors||[]).includes(sensor));
  if(proposal)station={...station,sensors:[...new Set([...(station.sensors||[]),...addedSensors])]};
  const box=document.createElement('div');box.className='station-popup';
  const title=document.createElement('strong');title.textContent=station.name?`${station.station_id} · ${station.name}`:station.station_id;
  const list=document.createElement('ul');
  for(const sensor of station.sensors||[]){
    const item=document.createElement('li');
    item.append(sensorDot(sensor),SENSORS[sensor]?.[0]||sensor);
    if(addedSensors.includes(sensor)){item.className='sensor-added';const added=document.createElement('small');added.textContent='Lagt til';item.append(added);}
    if((station.layers||[]).includes(sensor)){const used=document.createElement('small');item.append(used);}
    list.append(item);
  }
  if(!list.children.length){const item=document.createElement('li');item.textContent='Ingen kjente sensorer';list.append(item);}
  box.append(title,list);
  if(proposal){
    const actions=document.createElement('div');actions.className='popup-actions';
    //const edit=document.createElement('button');edit.type='button';edit.textContent='Endre nye sensorer';edit.onclick=()=>promptNewStation(latlng,proposal);
    //const remove=document.createElement('button');remove.type='button';remove.textContent='Fjern nye sensorer';remove.onclick=()=>{map.closePopup();requestState({action:'remove',id:proposal.id});};
    //actions.append(edit,remove);box.append(actions);
  }
  if(station.kind==='manual'){
    const actions=document.createElement('div');actions.className='popup-actions';
    const edit=document.createElement('button');edit.type='button';edit.textContent='Endre sensorer';
    edit.onclick=()=>promptNewStation(latlng,station);
    const remove=document.createElement('button');remove.type='button';remove.textContent='Fjern målepunkt';
    remove.onclick=()=>{map.closePopup();requestState({action:'remove',id:station.id});};
    actions.append(edit,remove);box.append(actions);
  }
  if(station.svv!==undefined){
    const owners=document.createElement('div');
    owners.textContent='Eier: '+((station.owners||[]).join(', ')||'Ikke oppgitt');
    box.append(owners);
    const svv=document.createElement('div');
    svv.textContent=station.svv?'Vegværstasjon':'Ikke vegværstasjon';
    if(station.svv)svv.style.fontWeight='600';
    box.append(svv);
  }
  if(selectionKind){
    const actions=document.createElement('div');actions.className='popup-actions';
    const choose=document.createElement('button');choose.type='button';choose.textContent='Velg stasjon';
    choose.onclick=()=>{map.closePopup();selectStation(selectionKind,selectionIndex);};
    actions.append(choose);box.append(actions);
  }
  if(station.kind==='observation'&&state?.network==='frost'&&state.frost_ready){
    const actions=document.createElement('div');actions.className='popup-actions';
    if(station.removed){
      const note=document.createElement('small');note.textContent='Fjernet fra scenariet';actions.append(note);
    }else{
      if(!(station.sensors||[]).includes('temperature')){
        const addTemperature=document.createElement('button');addTemperature.type='button';addTemperature.textContent='Legg til temperatursensor';
        addTemperature.onclick=()=>{map.closePopup();requestState({action:'add_sensor',station_id:station.station_id,sensors:['temperature']});};
        actions.append(addTemperature);
      }else if(addedSensors.includes('temperature')){
        const removeTemperature=document.createElement('button');removeTemperature.type='button';removeTemperature.textContent='Fjern temperatursensor';
        removeTemperature.onclick=()=>{map.closePopup();requestState({action:'remove_sensor',station_id:station.station_id,sensor:'temperature'});};
        actions.append(removeTemperature);
      }
      // const hint=document.createElement('small');hint.textContent=element('station-move-mode').checked?
      //   'Dra markøren for å flytte hele stasjonen med alle sensorer':
      //   'Slå på Flyttemodus for å flytte stasjonen';
      const remove=document.createElement('button');remove.type='button';remove.textContent='Fjern målepunkt';
      remove.onclick=()=>{map.closePopup();requestState({action:'remove_frost',station_id:station.station_id});};
      actions.append(remove);
      // actions.append(hint,remove);
    }
    if(station.removed||station.original_position||station.sensors_changed){
      const restore=document.createElement('button');restore.type='button';restore.textContent='Gjenopprett stasjon';
      restore.onclick=()=>{map.closePopup();requestState({action:'restore_frost',station_id:station.station_id});};
      actions.append(restore);
    }
    box.append(actions);
  }
  L.popup({autoPan:true}).setLatLng(latlng).setContent(box).openOn(map);
}
function promptNewStation(latlng,station=null){
  const options=state?.layer_options||[];
  if(!options.length){if(!station)requestState({action:'add',lat:latlng.lat,lon:latlng.lng});return;}
  const form=document.createElement('form');form.className='add-popup';
  const title=document.createElement('strong');title.textContent=station?'Endre sensorer':'Nytt målepunkt · velg sensorer';form.append(title);
  for(const option of options){
    const label=document.createElement('label'),input=document.createElement('input');
    input.type='checkbox';input.value=option.phenomenon;input.checked=station?(station.sensors||[]).includes(option.phenomenon):state.layers.includes(option.dataset);
    label.append(input,sensorDot(option.phenomenon),SENSORS[option.phenomenon]?.[0]||option.phenomenon);form.append(label);
  }
  const actions=document.createElement('div');actions.className='popup-actions';
  const cancel=document.createElement('button');cancel.type='button';cancel.textContent='Avbryt';cancel.onclick=()=>map.closePopup();
  const confirm=document.createElement('button');confirm.type='submit';confirm.textContent=station?'Lagre':'Bekreft';
  actions.append(cancel,confirm);form.append(actions);
  form.onsubmit=event=>{
    event.preventDefault();
    const sensors=[...form.querySelectorAll('input:checked')].map(input=>input.value);
    if(!sensors.length){error('Velg minst én sensor');return;}
    map.closePopup();requestState(station?{action:'sensors',id:station.id,sensors}:{action:'add',lat:latlng.lat,lon:latlng.lng,sensors});
  };
  L.popup({closeOnClick:false,autoPan:true}).setLatLng(latlng).setContent(form).openOn(map);
}
element('comparison-baseline').onclick=()=>requestState({
  action:'comparison',value:state.comparison_scope==='baseline'?'selected':'baseline'
});
element('map').addEventListener('mouseleave',clearPreview);
map.on('dragstart zoomstart',clearPreview);
map.on('click',async event=>{
  const canAddStation=Date.now()-popupClosedAt>300;
  if(event.originalEvent?.ctrlKey||event.originalEvent?.metaKey)inspectKuldehull(event.latlng);
  else if(event.originalEvent?.altKey)inspectBestStation(event.latlng);
  else if(element('coverage-show').checked&&await inspectCoverage(event.latlng))return;
  else if(mode.startsWith('terrain:'))inspectTerrain(event.latlng);
  else if(mode==='climate')inspectClimate(event.latlng);
  else if(canAddStation)promptNewStation(event.latlng);
});
let popupClosedAt=0;
map.on('popupclose',()=>{popupClosedAt=Date.now();});
element('help-open').onclick=()=>element('help-dialog').showModal();
element('help-close').onclick=()=>element('help-dialog').close();
element('help-dialog').onclick=event=>{
  if(event.target===element('help-dialog'))element('help-dialog').close();
};
element('help-dialog').onkeydown=event=>{
  if(event.key==='Escape'){
    event.preventDefault();
    event.stopPropagation();
    element('help-dialog').close();
  }
};
element('station-visibility').onclick=()=>{
  stationsVisible=!stationsVisible;clearPreview();
  for(const layer of [fixedMarkers,manualMarkers]){if(stationsVisible)layer.addTo(map);else map.removeLayer(layer);}
  const button=element('station-visibility'),label=stationsVisible?'Skjul stasjoner':'Vis stasjoner';
  button.setAttribute('aria-pressed',String(stationsVisible));button.setAttribute('aria-label',label);button.title=label;
  button.textContent=label;
  renderCalculationPoints();
};
element('station-calculation-points').onchange=renderCalculationPoints;
element('station-move-mode').checked=false;
element('station-move-mode').onchange=()=>{clearPreview();if(state)render();};
element('phenomena').addEventListener('change',event=>{
  if(!event.target.matches('input[data-phenomenon]'))return;
  const checked=[...element('phenomena').querySelectorAll('input[data-phenomenon]:checked')].map(input=>input.value);
  requestState({action:'layers',value:checked});
});
element('network').onchange=()=>{
  if(state.stations.length&&!window.confirm('Bytte værstasjoner og fjerne alle egne forslag?')){element('network').value=selectedNetwork();return;}
  requestState({action:'network',value:element('network').value,pross_only:['svv','svv_met'].includes(element('network').value)});
};
let stationDatePreview=null,stationDateSerial=0;
function cancelStationDate(){
  stationDateSerial++;
  stationDatePreview=null;
  element('station-date-preview').hidden=true;
  element('station-date-apply').disabled=true;
  element('station-since').value=state?.station_since||'';
}
element('station-since').onchange=async()=>{
  const value=element('station-since').value,serial=++stationDateSerial;
  stationDatePreview=null;
  element('station-date-preview').hidden=false;
  element('station-date-apply').disabled=true;
  element('station-date-counts').textContent='Teller stasjoner uten å beregne kartet…';
  try{
    const response=await fetch('/api/station-date-preview',{method:'POST',
      headers:{'Content-Type':'application/json'},body:JSON.stringify({value})});
    const result=await response.json();
    if(!response.ok)throw new Error(result.error||'Stasjonene kunne ikke telles');
    if(serial!==stationDateSerial||result.revision!==state.revision)return;
    stationDatePreview=result;
    element('station-date-counts').replaceChildren();
    for(const layer of result.layers){
      const line=document.createElement('div');
      line.textContent=`${layer.label}: ${number(layer.removed,0)} av ${number(layer.total,0)} utelates av datoen (${number(layer.unknown,0)} med ukjent dato). ${number(layer.remaining,0)} beholdes. Endring fra nå: −${number(layer.additional_removed,0)} / +${number(layer.restored,0)} stasjoner.`;
      element('station-date-counts').append(line);
    }
    element('station-date-apply').disabled=busy;
  }catch(problem){
    if(serial===stationDateSerial){cancelStationDate();error(problem.message);}
  }
};
element('station-date-cancel').onclick=cancelStationDate;
element('station-date-apply').onclick=()=>{
  if(!stationDatePreview||busy)return;
  const preview=stationDatePreview;
  requestState({action:'station_since',value:preview.value,expected_revision:preview.revision});
};

let countyOptions=[],countyTimer=null;
async function commitPendingCountySelection(){
  clearTimeout(countyTimer);
  if(!countyOptions.length)return true;
  const selected=[...document.querySelectorAll('#county-options input:checked')].map(input=>input.value);
  const saved=state.counties??countyOptions.map(item=>item.code);
  if(selected.length===saved.length&&selected.every(code=>saved.includes(code)))return true;
  return await requestState({action:'counties',value:selected});
}
function renderCountySelection(){
  if(!state||!countyOptions.length)return;
  const selected=new Set(state.counties??countyOptions.map(item=>item.code));
  document.querySelectorAll('#county-options input').forEach(input=>{input.checked=selected.has(input.value);input.disabled=busy;});
  element('county-count').textContent=selected.size===countyOptions.length?'Alle valgt':`${selected.size} av ${countyOptions.length} valgt`;
}
function applyCountySelection(){
  clearTimeout(countyTimer);
  const selected=[...document.querySelectorAll('#county-options input:checked')].map(input=>input.value);
  element('county-count').textContent=selected.length===countyOptions.length?'Alle valgt':`${selected.length} av ${countyOptions.length} valgt`;
  countyTimer=setTimeout(()=>requestState({action:'counties',value:selected}),350);
}
for(const action of ['all','none'])element(`county-${action}`).onclick=()=>{
  document.querySelectorAll('#county-options input').forEach(input=>input.checked=action==='all');applyCountySelection();
};
fetch('/api/counties').then(async response=>{
  const result=await response.json();if(!response.ok)throw new Error(result.error||'Fylker kunne ikke lastes');
  countyOptions=result.counties;
  for(const county of countyOptions){
    const label=document.createElement('label'),input=document.createElement('input');
    input.type='checkbox';input.value=county.code;input.checked=true;input.onchange=applyCountySelection;
    label.style.cssText='display:flex;align-items:center;gap:6px;font-size:12px';
    label.title=county.name;
    label.append(input,document.createTextNode(county.name.split(' - ')[0]));element('county-options').append(label);
  }
  renderCountySelection();
}).catch(failure=>{element('county-error').textContent=failure.message;});
function resetViewDefaults(){
  mode='uncertainty';cutoffRange=null;rmseRangeKey=null;overlayKey=null;
  for(const id of ['road-score-4','road-score-5','road-corridor','kuldehull-show','kuldehull-only','coverage-show'])element(id).checked=false;
  element('road-show').checked=true;
  for(const id of ['road-distance','road-reference','kuldehull-depth','opacity','rmse-min'])element(id).value=element(id).defaultValue;
  element('road-categories').querySelectorAll('input').forEach(input=>input.checked=['E','R','F'].includes(input.value));
  element('coverage-classes').querySelectorAll('input').forEach(input=>input.checked=true);
  if(!stationsVisible)element('station-visibility').click();
  document.querySelectorAll('aside details').forEach(details=>details.open=details.id==='station-picker');
  element('station-move-mode').checked=false;
  render();map.fitBounds(state.bounds,{padding:[8,8]});
}
element('reset-defaults').onclick=async()=>{
  if(busy||!state||!window.confirm('Tilbakestille alle stasjoner, filtre og kartinnstillinger til standard?'))return;
  if(await requestState({action:'reset_defaults'}))resetViewDefaults();
};
element('undo').onclick=()=>requestState({action:'undo'});
element('reset').onclick=()=>{if(window.confirm('Fjerne alle egne stasjonsforslag?'))requestState({action:'reset'});};
element('fit').onclick=()=>{if(state)map.fitBounds(state.bounds,{padding:[10,10]});};
element('opacity').oninput=()=>{overlay?.setOpacity(overlayOpacity());};
element('export').onclick=async event=>{
  event.preventDefault();
  if(element('export').getAttribute('aria-disabled')==='true')return;
  const link=element('export');link.setAttribute('aria-disabled','true');
  try{
    const response=await fetch('/api/export.gpkg',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({mode,revision:state.revision,rmse_min:rmseMinimum(),roads:getWeatherRoadSettings(),kuldehull:getKuldehullSettings()})});
    if(!response.ok){const result=await response.json();throw new Error(result.error||'GeoPackage kunne ikke eksporteres');}
    const url=URL.createObjectURL(await response.blob()),download=document.createElement('a');
    download.href=url;download.download='usikkerhetskartlag.gpkg';download.click();setTimeout(()=>URL.revokeObjectURL(url),60000);
  }catch(failure){error(failure.message);}finally{controls();}
};
let cutoffRange=null;
const cutoffNumber=new Intl.NumberFormat('nb-NO',{maximumFractionDigits:3});
function rmseMinimum(){
  if(mode==='reduction'||mode==='climate'||mode.startsWith('terrain:'))return 0;
  const percentile=Number(element('rmse-min').value);
  return percentile&&cutoffRange?.cells?Math.max(Number.MIN_VALUE,cutoffRange.cutoffs[percentile]):0;
}
function renderUncertaintyCutoff(){
  const value=rmseMinimum(),swatch=element('rmse-min-value');
  swatch.textContent='';
  const percentile=Number(element('rmse-min').value),combined=state.layers.length>1;
  const unit=combined?'':mode==='placement'?' °C·km²':` ${state.unit||'°C'}`;
  const cut=value?cutoffRange.cutoffs[percentile]:0;
  const label=mode==='reduction'?'Ikke tilgjengelig for Endring i usikkerhet':value?
    `${cutoffNumber.format(cut)}${unit} · ${number(cutoffRange.visible_percent[percentile],1)} % av rutene vises`:'Av – alle farger vises';
  swatch.style.background=value&&!combined&&mode!=='placement'?roadErrorColour(cut):'transparent';
  element('rmse-cutoff-text').textContent=label;
  //swatch.title=label;swatch.setAttribute('aria-label',label);
  element('rmse-min').setAttribute('aria-valuetext',label);
}
let rmseRangeKey=null,rmseRangeSerial=0;
async function refreshRmseRange(){
  const slider=element('rmse-min');
  if((!['RMSE','P95'].includes(state.metric)&&state.layers.length<2)||mode.startsWith('terrain:')||mode==='climate'||mode==='reduction'){
    slider.disabled=true;cutoffRange=null;rmseRangeKey=null;rmseRangeSerial++;return;
  }
  const settings=getWeatherRoadSettings(),rangeMode=mode;
  const placement=mode==='placement',combined=state.layers.length>1;
  if(placement&&!placementCatalogs.get(JSON.stringify([state.dataset,selectedPlacementNetwork()]))){
    slider.disabled=true;cutoffRange=null;rmseRangeKey=null;rmseRangeSerial++;return;
  }
  const key=JSON.stringify([state.dataset,state.layers,state.combine,state.cell_m,state.revision,rangeMode,settings]);
  if(key===rmseRangeKey)return;
  rmseRangeKey=key;slider.disabled=true;cutoffRange=null;
  const serial=++rmseRangeSerial;
  try{
    const response=await (combined?fetch(`/api/combined-range?mode=${rangeMode}`):fetch('/api/rmse-range',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({mode:rangeMode,roads:settings})}));
    const result=await response.json();
    if(!response.ok)throw new Error(result.error||'Filtergrensen kunne ikke lastes');
    if(serial!==rmseRangeSerial)return;
    cutoffRange=result;
    slider.disabled=!result.cells;
    renderUncertaintyCutoff();
    if(rmseMinimum()){repaintMapRaster();updateMap();}
  }catch(failure){if(serial===rmseRangeSerial){rmseRangeKey=null;error(failure.message);}}
}
let rmseTimer=null;
element('rmse-min').oninput=()=>{
  renderUncertaintyCutoff();
  clearPreview();clearTimeout(rmseTimer);
  if(mode==='placement'&&state.layers.length===1&&placementTileLayer||
      ['uncertainty','baseline'].includes(mode)&&(uncertaintyTileLayer||overlay?.rasterCanvas||
        activeMapImage?.rasterCanvas||pendingMapImage?.numeric)){
    repaintMapRaster();return;
  }
  overlaySerial++;overlayKey=null;cancelQueuedMapImage();
  rmseTimer=setTimeout(updateMap,150);
};
document.querySelectorAll('[data-mode]').forEach(button=>button.onclick=()=>{mode=button.dataset.mode;updateMap();});
document.addEventListener('keydown',event=>{if((event.ctrlKey||event.metaKey)&&event.key==='z'&&!['INPUT','TEXTAREA','SELECT'].includes(document.activeElement.tagName)){event.preventDefault();requestState({action:'undo'});}});
new ResizeObserver(()=>map.invalidateSize()).observe(element('map'));
function showProgress(jobs){
  const status=state?.frost_precompute;
  element('precompute-status').hidden=!status;
  if(!status)return;
  const label={missing:'Mangler forhåndsberegning',ready:'Lagret og klar',failed:'Ugyldig lagret flate'};
  const done=jobs.filter(job=>job.state==='ready').length;
  element('precompute-current').textContent=`Eksakt Frost-grunnflate: ${label[status.state]||status.state}${status.error?` · ${status.error}`:''}`;
  element('precompute-summary').textContent=`${done} av ${jobs.length} innlastede grunnflater klare`;
  element('precompute-jobs').replaceChildren();
  for(const job of jobs){const item=document.createElement('li');item.textContent=`${number(job.cell_m,0)} m · ${job.label} · ${label[job.state]||job.state}`;element('precompute-jobs').append(item);}
}
async function refreshProgress(){
  if(!state||busy||progressBusy)return;
  progressBusy=true;
  const revision=state.revision;
  try{
    const response=await fetch('/api/precompute');
    if(!response.ok)throw new Error('Beregningsstatus kunne ikke hentes');
    const result=await response.json();
    const calculation=result.calculation,progress=element('calculation-progress');
    progress.hidden=!calculation;
    if(calculation){
      const remaining=calculation.remaining_seconds;
      const eta=remaining===null?'Tidsanslag kommer når fremdriften er målt':`Anslått gjenstår i denne fasen: ${Math.max(1,Math.ceil(remaining/60))} min`;
      progress.textContent=`${calculation.title||''} · ${calculation.stage}${calculation.step?` · steg ${calculation.step}/${calculation.steps}`:''}${calculation.total?` · ${calculation.completed}/${calculation.total}`:''}${calculation.state==='running'?` · ${calculation.stale?'Ingen ny fremdrift meldt nylig':eta}`:''}${calculation.error?` · ${calculation.error}`:''}`;
    }
    if(busy||state.revision!==revision)return;
    const previous=state.frost_precompute;
    const matching=result.jobs.find(job=>job.cell_m===state.cell_m&&job.label===`${state.weather_label} · ${state.metric}`);
    state.precompute_jobs=result.jobs;
    if(matching)state.frost_precompute=matching;
    showProgress(result.jobs);
    if(previous?.state!=='ready'&&matching?.state==='ready')await requestState();
  }catch(failure){error(failure.message);}finally{progressBusy=false;}
}
setInterval(refreshProgress,5000);
element('coverage-show').onchange=scheduleCoverage;
initializeCoverage();
requestState().then(()=>{if(state)map.fitBounds(state.bounds,{padding:[8,8]});});

function selectedRoadScoreClasses(){
  return roadOptions?.score_network?['4','5'].filter(code=>
    roadOptions.classes.includes(code)&&element(`road-score-${code}`).checked):[];
}

function getRoadSettings(includeHidden=false){
  const classes=selectedRoadScoreClasses();
  if(!roadOptions?.available||!includeHidden&&!classes.length)return null;
  const bounds=map.getBounds();
  return {bounds:[Math.max(-180,bounds.getWest()),Math.max(-85,bounds.getSouth()),Math.min(180,bounds.getEast()),Math.min(85,bounds.getNorth())],
    categories:Array.from(element('road-categories').querySelectorAll('input:checked')).map(input=>input.value)
      .filter(code=>classes.length||['E','R','F'].includes(code)),
    classes,
    reference:element('road-reference').value.trim(),zoom:map.getZoom(),colour:'score',geometry_only:true,
    corridor:false};
}

function getWeatherRoadSettings(includeHidden=false){
  if(!includeHidden&&!element('road-corridor').checked||!roadOptions?.available)return null;
  const bounds=map.getBounds();
  return {bounds:[Math.max(-180,bounds.getWest()),Math.max(-85,bounds.getSouth()),Math.min(180,bounds.getEast()),Math.min(85,bounds.getNorth())],
    categories:roadOptions.categories.map(item=>item.code),classes:roadOptions.classes,
    reference:'',corridor:true,distance_m:Number(element('road-distance').value)};
}

function displayColourLimits(){
  return [state.color_min,state.color_max];
}

function roadErrorColour(value){
  if(value===null||!Number.isFinite(value)||!state)return '#7b8480';
  if(usesFixedP95Scale())return uncertaintyScaleColour(value<=1?value/2:.5+(value-1)/4);
  const [minimum,maximum]=displayColourLimits();
  return uncertaintyScaleColour(Math.log(Math.max(value,minimum)/minimum)/Math.log(maximum/minimum));
}

function uncertaintyScaleColour(position){
  const stops=[[5,48,97],[107,172,209],[247,247,247],[228,128,101],[103,0,31]];
  const scaled=Math.max(0,Math.min(1,position))*4;
  const left=Math.min(3,Math.floor(scaled)),fraction=scaled-left;
  const color=stops[left].map((channel,index)=>Math.round(channel+(stops[left+1][index]-channel)*fraction));
  return `rgb(${color.join(',')})`;
}

function placementScaleColour(position,combined){
  const stops=combined?[[255,255,217],[236,247,177],[198,232,180],[126,204,187],[64,181,195],[29,144,191],[34,93,167],[36,51,146],[8,29,88]]:
    [[255,255,204],[254,236,159],[254,216,117],[253,177,75],[252,140,59],[251,76,41],[226,25,28],[187,0,38],[128,0,38]];
  const scaled=Math.max(0,Math.min(1,position))*(stops.length-1);
  const left=Math.min(stops.length-2,Math.floor(scaled)),fraction=scaled-left;
  return `rgb(${stops[left].map((channel,index)=>Math.round(channel+(stops[left+1][index]-channel)*fraction)).join(',')})`;
}

function roadStyle(feature){
  const scoreClass=Number(feature.properties.VS_Total_klasse);
  return {color:ROAD_SCORE_COLOURS[scoreClass-1]||'#7b8480',
    weight:feature.properties.overview?1.6:2.5,opacity:.95,
    dashArray:feature.properties.isferry?'5 5':null};
}

function scheduleRoads(){
  clearTimeout(roadTimer);
  roadSerial++;
  updateBackgroundRoads();
  if(!selectedRoadScoreClasses().length){roadPending=false;roadController?.abort();roadLines.clearLayers();roadCasing.clearLayers();roadResult=null;roadRequestKey=null;renderRoadSummary();return;}
  if(roadOptions)roadTimer=setTimeout(refreshRoads,450);
}

function updateBackgroundRoads(){
  const scores=selectedRoadScoreClasses().length>0;
  element('road-categories').querySelectorAll('input').forEach(input=>{
    input.disabled=!scores&&!['E','R','F'].includes(input.value);
  });
  if(!element('road-show').checked){map.removeLayer(backgroundRoads);return;}
  const categories=roadOptions?.available?
    [...element('road-categories').querySelectorAll('input:checked')].map(input=>input.value).filter(code=>['E','R','F'].includes(code)):
    ['E','R','F'];
  const query=new URLSearchParams({categories:JSON.stringify(categories),reference:element('road-reference').value.trim()});
  if(state?.counties!==null&&state?.counties!==undefined)query.set('counties',JSON.stringify(state.counties));
  const key=query.toString();
  if(key!==backgroundRoadKey){
    backgroundRoadKey=key;backgroundRoadError=false;
    backgroundRoads.setUrl(`/api/roads/background/{z}/{x}/{y}.png?${key}`);
  }
  if(!map.hasLayer(backgroundRoads))backgroundRoads.addTo(map);
}

async function refreshRoads(){
  if(!state||!selectedRoadScoreClasses().length)return;
  if(roadBusy){roadPending=true;return;}
  const serial=roadSerial;
  const settings=getRoadSettings();
  if(!settings){roadLines.clearLayers();roadCasing.clearLayers();roadResult=null;renderRoadSummary();return;}
  const {corridor,distance_m,...displaySettings}=settings;
  const requestKey=JSON.stringify([displaySettings,state.counties]);
  if(roadResult&&roadRequestKey===requestKey){renderRoadSummary();return;}
  roadBusy=true;
  roadController=new AbortController();
  roadStatus();
  try{
    const response=await fetch('/api/roads/view',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(displaySettings),signal:roadController.signal});
    const result=await response.json();
    if(!response.ok)throw new Error(result.error||'Vegnettet kunne ikke lastes');
    if(serial!==roadSerial)return;
    result.colourKey=JSON.stringify(settings);
    roadResult=result;
    roadRequestKey=requestKey;
    roadLines.clearLayers();roadCasing.clearLayers();roadCasing.addData(result);roadLines.addData(result);renderRoadSummary();
  }catch(failure){if(failure.name!=='AbortError'&&serial===roadSerial)roadStatus(failure.message);}
  finally{roadBusy=false;if(roadPending){roadPending=false;refreshRoads();}}
}

function renderRoadSummary(){
  const summary=roadResult?.summary;
  element('road-download').disabled=!roadResult?.features.length;
  roadStatus(!roadOptions||roadOptions.available?
    (summary?.truncated?'Ikke alle veger kan vises. Zoom inn for å se flere.':''):'Vegnettfilen er ikke tilgjengelig');
}

function roadStatus(message=''){
  const status=element('road-status');
  status.textContent=message;status.hidden=!message;status.classList.toggle('warning',Boolean(message));
}

function showRoadPopup(feature,location){
  const properties=feature.properties,popup=document.createElement('div');popup.className='road-popup';
  const title=document.createElement('strong');title.textContent=`${properties.roadid||'Veg uten nummer'}`;
  const values=document.createElement('dl');
  const rows=[...(roadOptions?.score_network?[['Vegscore',properties.VS_Total],['Vegscoreklasse',displayRoadScoreClass(properties.VS_Total_klasse)]]:[['Vegklasse',properties.roadclass]]),['Vegtype',properties.roadtype],
    ['Lenkelengde',`${number(properties.displayed_km,2)} km`]];
  if(properties.istunnel)rows.push(['Tunnel','Ja']);if(properties.isbridge)rows.push(['Bru','Ja']);if(properties.isferry)rows.push(['Ferje','Ja']);
  for(const [name,value]of rows){const key=document.createElement('dt'),text=document.createElement('dd');key.textContent=name;text.textContent=value??'Ukjent';values.append(key,text);}
  const add=document.createElement('button');add.innerHTML='<i data-lucide="map-pin-plus"></i>';add.append(document.createTextNode('Foreslå stasjon her'));add.onclick=()=>{map.closePopup();promptNewStation(location);};
  const terrain=document.createElement('button');terrain.className='secondary';terrain.innerHTML='<i data-lucide="mountain"></i>';terrain.append(document.createTextNode('Vis terrengverdier'));terrain.onclick=()=>inspectTerrain(location);
  const components=document.createElement('div');components.className='score-components';
  popup.append(title,values);
  if(roadOptions?.score_network)popup.append(components);
  popup.append(add,terrain);
  const leafletPopup=L.popup({maxWidth:340}).setLatLng(location).setContent(popup).openOn(map);icons();
  if(roadOptions?.score_network)loadRoadComponents(feature,location,components,title,leafletPopup);
}

async function loadRoadComponents(feature,location,container,title,popup){
  container.textContent='Laster scorefordeling…';
  try{
    const settings=getRoadSettings()||{};
    const selection=Number.isInteger(feature.properties.fid)?{fid:feature.properties.fid}:{lon:location.lng,lat:location.lat};
    const response=await fetch('/api/roads/components',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({...settings,...selection})});
    const result=await response.json();
    if(!response.ok)throw new Error(result.error||'Scorefordelingen kunne ikke hentes');
    container.replaceChildren();
    if(!Number.isInteger(feature.properties.fid))title.textContent=`${result.roadid}`;
    const heading=document.createElement('strong');heading.textContent='Scorefordeling';
    const bar=document.createElement('div');bar.className='score-bar';bar.setAttribute('role','img');
    bar.setAttribute('aria-label','Relativ fordeling mellom scorekomponentene');
    const key=document.createElement('div');key.className='score-key';
    for(const component of result.components){
      if(component.share>0){
        const segment=document.createElement('span');segment.className='score-segment';
        segment.style.width=`${component.share*100}%`;segment.style.background=component.colour;
        segment.title=component.label;bar.append(segment);
      }
      const label=document.createElement('span'),dot=document.createElement('i');dot.style.background=component.colour;
      dot.setAttribute('aria-hidden','true');
      label.append(dot,document.createTextNode(component.label+(component.share===null?' · mangler':'')));key.append(label);
    }

    container.append(heading,bar,key);
    if(map.hasLayer(popup))popup.update();
  }catch(failure){
    container.textContent=failure.message;
    if(map.hasLayer(popup))popup.update();
  }
}


function changeRoadFilters(){clearPreview();scheduleRoads();if(state)updateMap();}

async function initializeRoads(){
  roadStatus();
  try{
    const response=await fetch('/api/roads/options');if(!response.ok)throw new Error('Vegnettets metadata kunne ikke lastes');roadOptions=await response.json();
    for(const code of ['4','5']){
      const input=element(`road-score-${code}`);
      input.disabled=!roadOptions.available||!roadOptions.score_network||!roadOptions.classes.includes(code);
      if(input.disabled)input.checked=false;
    }
    if(!roadOptions.available){scheduleRoads();return;}
    element('road-categories').replaceChildren();
    for(const category of roadOptions.categories){
      const label=document.createElement('label'),input=document.createElement('input');input.type='checkbox';input.value=category.code;input.checked=['E','R','F'].includes(category.code);input.onchange=changeRoadFilters;
      //label.title=`${number(category.links,0)} lenker i datakilden`;
      label.append(input,document.createTextNode(`${category.label}`));
      element('road-categories').append(label);
    }
    element('road-show').disabled=false;scheduleRoads();
  }catch(failure){roadStatus(failure.message);}
}

element('road-show').onchange=changeRoadFilters;
for(const id of ['road-score-4','road-score-5'])element(id).onchange=()=>{
  roadController?.abort();
  roadLines.clearLayers();roadCasing.clearLayers();roadResult=null;roadRequestKey=null;
  changeRoadFilters();
};
element('road-corridor').onchange=async()=>{
  if(element('road-corridor').checked){
    if(!roadOptions){
      if(!roadInitialization)roadInitialization=initializeRoads().finally(()=>{roadInitialization=null;});
      await roadInitialization;
    }
    if(!roadOptions?.available){element('road-corridor').checked=false;error('Vegnettet er ikke tilgjengelig');}
  }
  changeRoadFilters();
};
element('road-distance').onchange=changeRoadFilters;
element('road-reference').onchange=changeRoadFilters;
element('road-reference').onkeydown=event=>{if(event.key==='Enter'){event.preventDefault();event.target.blur();}};
element('road-picker').addEventListener('toggle',async()=>{
  if(element('road-picker').open&&!roadOptions&&!roadInitialization){
    roadInitialization=initializeRoads().finally(()=>{roadInitialization=null;});
    await roadInitialization;
  }
});
element('road-download').onclick=()=>{
  if(!roadResult)return;
  const data={...roadResult,dataset:state.dataset,network:state.network,threshold:state.threshold,selected_station:state.selected_station,
    model_warning:state.model_warning,training_period:state.training_period,
    source_layer:roadOptions.layer,features:roadResult.features.map(feature=>({...feature,properties:{...feature.properties,
      weather_dataset:state.dataset,weather_unit:roadResult.unit,target_cell_m:roadResult.cell_m}}))};
  const link=document.createElement('a'),url=URL.createObjectURL(new Blob([JSON.stringify(data)],{type:'application/geo+json'}));
  link.href=url;link.download='vegnett_utsnitt.geojson';link.hidden=true;document.body.append(link);link.click();link.remove();
  setTimeout(()=>URL.revokeObjectURL(url),1000);
};
map.on('moveend',()=>{scheduleRoads();scheduleKuldehull();scheduleCoverage();if(state)updateMap();});

async function loadPlacementOptions(){
  const dataset=state.dataset;
  const key=JSON.stringify([dataset,selectedPlacementNetwork()]);
  if(placementPending.has(key))return;
  placementPending.add(key);
  try{
    const response=await fetch('/api/placement');const layer=await response.json();
    if(!response.ok)throw new Error(layer.error||'Plasseringskartet kunne ikke lastes');
    const returnedKey=JSON.stringify([layer.dataset,layer.station_network||selectedPlacementNetwork()]);
    placementCatalogs.set(returnedKey,layer);
    if(key===JSON.stringify([state.dataset,selectedPlacementNetwork()])&&mode==='placement')updateMap();
  }catch(failure){if(key===JSON.stringify([state.dataset,selectedPlacementNetwork()])&&mode==='placement'){error(failure.message);mode='uncertainty';updateMap();}}
  finally{placementPending.delete(key);}
}

function populateTerrainOptions(){
  if(!state)return;
  const catalog=terrainCatalogs.get(state.cell_m);if(!catalog)return;
  const select=element('terrain-layer');select.replaceChildren();
  const weather=document.createElement('option');weather.value='';weather.textContent='Usikkerhetskart';select.append(weather);
  if(catalog.placement_available&&['temperature:senorge30','temperature:direct_p95'].includes(state.dataset)){const option=document.createElement('option');option.value='placement';option.textContent='Plasseringsgevinst';select.append(option);}
  if(climateCatalog){const option=document.createElement('option');option.value='climate';option.textContent='Vegklimasonekart';select.append(option);}
  for(const layer of catalog.layers.filter(layer=>layer.field==='elevation')){const option=document.createElement('option');option.value=layer.field;option.textContent='Høyde over havet';select.append(option);}
  select.value=mode==='climate'?'climate':mode==='placement'?'placement':mode.startsWith('terrain:')?mode.split(':')[1]:'';controls();
}

async function loadTerrainOptions(){
  if(!state)return;
  const spacing=state.cell_m;
  if(terrainCatalogs.has(spacing)){populateTerrainOptions();return;}
  if(terrainPending.has(spacing))return;
  terrainPending.add(spacing);
  try{
    const response=await fetch('/api/terrain/options');const catalog=await response.json();
    if(!response.ok)throw new Error(catalog.error||'Terrenglagene kunne ikke lastes');
    terrainCatalogs.set(catalog.cell_m,catalog);
    if(state.cell_m===catalog.cell_m){populateTerrainOptions();if(mode.startsWith('terrain:'))updateMap();}
  }catch(failure){error(failure.message);}finally{terrainPending.delete(spacing);}
}

async function inspectKuldehull(location){
  if(!state||busy||!kuldehullAvailable)return;
  clearPreview();
  const serial=++inspectionSerial;
  const content=document.createElement('div');content.textContent='Laster kuldehull…';
  const popup=L.popup({maxWidth:300}).setLatLng(location).setContent(content).openOn(map);
  try{
    const query=new URLSearchParams({lon:location.lng,lat:location.lat,depth:getKuldehullSettings().depth});
    const response=await fetch(`/api/kuldehull/point?${query}`);
    const result=await response.json();
    if(serial!==inspectionSerial||!map.hasLayer(popup))return;
    if(!response.ok)throw new Error(result.error||'Kuldehull kunne ikke undersøkes');
    content.replaceChildren();
    const heading=document.createElement('strong');
    heading.textContent=result.found?`Kuldehull`:'Ingen kuldehull her med valgt dybdefilter';
    content.append(heading);
    if(result.found){
      const entries=[['Maksdybde',result.max_depth_m,'m'],['Middeldybde',result.mean_depth_m,'m'],
        ['Areal',result.area_m2/1e6,'km²'],
        ['Volum',result.volume_m3,'m³']];
      for(const [label,value,unit] of entries){
        if(!Number.isFinite(value))continue;
        const row=document.createElement('div');row.textContent=`${label}: ${shortNumber(value)} ${unit}`;content.append(row);
      }
    }
    popup.setContent(content);
  }catch(failure){if(serial===inspectionSerial)content.textContent=failure.message;}
}

function getKuldehullSettings(){
  const view=map.getBounds(),size=map.getSize(),scale=Math.min(1,1600/Math.max(size.x,size.y));
  return {only:kuldehullAvailable&&element('kuldehull-only').checked,depth:Number(element('kuldehull-depth').value)||0,
    bounds:[Math.max(-180,view.getWest()),Math.max(-85,view.getSouth()),Math.min(180,view.getEast()),Math.min(85,view.getNorth())],
    width:Math.max(1,Math.round(size.x*scale)),height:Math.max(1,Math.round(size.y*scale))};
}

function scheduleKuldehull(){
  if(!kuldehullAvailable||!element('kuldehull-show').checked){
    clearTimeout(kuldehullTimer);
    kuldehullSerial++;kuldehullController?.abort();kuldehullRequestKey=null;
    if(kuldehullOverlay)map.removeLayer(kuldehullOverlay);
    if(kuldehullUrl)URL.revokeObjectURL(kuldehullUrl);
    kuldehullUrl=null;element('kuldehull-status').textContent='';
    kuldehullOverlay=null;return;
  }
  const settings=getKuldehullSettings();
  const key=JSON.stringify([settings.bounds,settings.width,settings.height,settings.depth,state?.counties]);
  if(key===kuldehullRequestKey)return;
  clearTimeout(kuldehullTimer);
  kuldehullRequestKey=key;
  const serial=++kuldehullSerial;
  kuldehullController?.abort();
  kuldehullController=new AbortController();
  const controller=kuldehullController;
  kuldehullTimer=setTimeout(async()=>{
    const view=map.getBounds();
    const bounds=[Math.max(-180,view.getWest()),Math.max(-85,view.getSouth()),Math.min(180,view.getEast()),Math.min(85,view.getNorth())];
    const size=map.getSize(),scale=Math.min(1,1600/Math.max(size.x,size.y));
    const query=new URLSearchParams({bounds:JSON.stringify(bounds),depth:getKuldehullSettings().depth,width:Math.max(1,Math.round(size.x*scale)),height:Math.max(1,Math.round(size.y*scale))});
    element('kuldehull-status').textContent='Laster kuldehull…';
    try{
      const response=await fetch(`/api/kuldehull.png?${query}`,{signal:controller.signal});
      if(!response.ok)throw new Error('Kuldehull kunne ikke lastes');
      const blob=await response.blob();
      if(serial!==kuldehullSerial)return;
      const url=URL.createObjectURL(blob);
      const next=L.imageOverlay(url,[[bounds[1],bounds[0]],[bounds[3],bounds[2]]],{opacity:0,interactive:false,zIndex:420}).addTo(map);
      next.once('load',()=>{
        if(serial!==kuldehullSerial){map.removeLayer(next);URL.revokeObjectURL(url);return;}
        if(kuldehullOverlay)map.removeLayer(kuldehullOverlay);
        if(kuldehullUrl)URL.revokeObjectURL(kuldehullUrl);
        kuldehullUrl=url;kuldehullOverlay=next;next.setOpacity(.75);element('kuldehull-status').textContent='';
      });
      next.once('error',()=>{map.removeLayer(next);URL.revokeObjectURL(url);if(serial===kuldehullSerial){kuldehullRequestKey=null;element('kuldehull-status').textContent='Kuldehull kunne ikke lastes';}});
    }catch(failure){
      if(serial===kuldehullSerial&&failure.name!=='AbortError'){kuldehullRequestKey=null;element('kuldehull-status').textContent='Kuldehull kunne ikke lastes';}
    }
  },150);
}

for(const id of ['kuldehull-show','kuldehull-only','kuldehull-depth'])element(id).onchange=()=>{
  const input=element('kuldehull-depth');
  if(!input.checkValidity()||input.value===''){input.reportValidity();return;}
  map.closePopup();clearPreview();overlayKey=null;scheduleKuldehull();if(state)updateMap();
};
fetch('/api/kuldehull/options').then(response=>response.json()).then(options=>{
  kuldehullAvailable=Boolean(options.available);
  element('kuldehull-status').textContent=kuldehullAvailable?'':'Kuldehull-laget mangler';
  controls();
}).catch(()=>{element('kuldehull-status').textContent='Kuldehull-laget er utilgjengelig';});

element('terrain-layer').onchange=()=>{
  const field=element('terrain-layer').value;mode=field==='climate'?'climate':field==='placement'?'placement':field?`terrain:${field}`:'uncertainty';map.closePopup();updateMap();
};

fetch('/api/climate/options').then(async response=>{
  const result=await response.json();
  if(!response.ok)throw new Error(result.error||'Klimasoner kunne ikke lastes');
  climateCatalog=result;populateTerrainOptions();if(mode==='climate')updateMap();
}).catch(problem=>error(problem.message));

function updateClimateMap(){
  if(!climateCatalog){overlaySerial++;overlayKey=null;cancelQueuedMapImage();overlay?.setOpacity(0);return;}
  element('terrain-layer').value='climate';
  element('legend-title').textContent=`Vegklimasoner`;
  committedLegendTitle=element('legend-title').textContent;
  element('domain').textContent='Norge / Klimasoner';document.title='Værnett | Klimasoner';
  const legend=element('climate-legend-items');legend.replaceChildren();
  for(const zone of climateCatalog.zones){
    const row=document.createElement('div'),swatch=document.createElement('span');
    swatch.style.cssText=`display:inline-block;width:10px;height:10px;margin-right:6px;background:${zone.color}`;
    row.append(swatch,document.createTextNode(zone.name));legend.append(row);
  }
  const size=map.getZoom()<=5?400:map.getZoom()<=7?800:1600;
  const roads=getWeatherRoadSettings(),query=roads?.corridor?`&roads=${encodeURIComponent(JSON.stringify(roads))}`:'';
  const key=JSON.stringify(['climate',state.counties,state.cell_m,size,query]);
  if(overlayKey===key&&(overlay||activeMapImage||pendingMapImage))return;
  overlayKey=key;const serial=++overlaySerial;
  queueMapImage(`/api/climate.png?size=${size}${query}`,climateCatalog.bounds,serial,'Laster klimakart','Klimasonene kunne ikke tegnes');
}

async function inspectClimate(location){
  if(busy)return;
  const serial=++inspectionSerial,content=document.createElement('div');
  content.textContent='Leser klimasone…';
  const popup=L.popup({maxWidth:300}).setLatLng(location).setContent(content).openOn(map);
  try{
    const response=await fetch(`/api/climate/point?lon=${location.lng}&lat=${location.lat}`);
    const result=await response.json();
    if(!response.ok)throw new Error(result.error||'Klimasonen kunne ikke leses');
    if(serial!==inspectionSerial)return;
    content.textContent=result.zone?result.zone.name:'Ingen klimasone i denne cellen';
    if(result.zone){
      const details=document.createElement('details'),heading=document.createElement('summary');
      heading.textContent='Gjennomsnittsverdier for sonen';
      heading.style.cursor='pointer';details.append(heading);content.append(details);
      details.addEventListener('toggle',()=>popup.update());
      heading.style.cssText='font-size:11px;font-weight:600;margin-top:8px;cursor:pointer';
      const values=document.createElement('dl');
      values.style.cssText='display:grid;grid-template-columns:1fr auto;gap:4px 12px;font-size:11px';
      const fields=[
        ['vintertemp','Vintertemperatur','°C'],
        ['sommertemp','Sommertemperatur','°C'],
        ['amplitude','Temperaturamplitude','°C'],
        ['vinternedbor','Vinternedbør','mm'],
        ['frostdager','Frostdager','dager'],
        ['hoyde','Høyde','m'],
        ['avstand_til_sjo','Avstand til sjø','km'],
        ['kystpavirkning','Kystpåvirkning','']
      ];
      for(const [field,label,unit] of fields){
        const value=result.statistics[field];if(value===undefined)continue;
        const name=document.createElement('dt'),data=document.createElement('dd');
        name.textContent=label;data.style.margin='0';
        data.textContent=`${number(value,field==='kystpavirkning'?3:2)}${unit?' '+unit:''}`;
        values.append(name,data);
      }
      details.append(values);
    }
    const note=document.createElement('div');note.className='cell-note';
    content.append(note);popup.update();
  }catch(problem){if(serial===inspectionSerial){content.textContent=problem.message;popup.update();}}
}

async function inspectBestStation(location){
  if(!state||busy)return;
  clearPreview();
  const revision=state.revision,serial=++inspectionSerial;
  const loading=document.createElement('span');loading.textContent='Finner beste stasjon her…';
  const popup=L.popup({maxWidth:340,maxHeight:Math.max(140,Math.min(380,map.getSize().y-120))}).setLatLng(location).setContent(loading).openOn(map);
  try{
    const response=await fetch(`/api/stations/best?lon=${encodeURIComponent(location.lng)}&lat=${encodeURIComponent(location.lat)}`);
    const result=await response.json();
    if(serial!==inspectionSerial||revision!==state.revision||!map.hasLayer(popup))return;
    if(!response.ok)throw new Error(result.error||'Beste stasjon kunne ikke beregnes');
    const stationPoint=L.latLng(result.best.lat,result.best.lon);
    L.polyline([location,stationPoint],{pane:'stations',color:'#075d96',weight:3,dashArray:'8 6',interactive:false}).addTo(inspectionMarkers);
    L.circleMarker(location,{pane:'stations',radius:7,color:'#fff',weight:2,fillColor:'#075d96',fillOpacity:1,interactive:false}).bindTooltip('Her',{permanent:true,direction:'bottom'}).addTo(inspectionMarkers);
    L.circleMarker(stationPoint,{pane:'stations',radius:13,color:'#edb138',weight:4,fillColor:'#075d96',fillOpacity:1,interactive:false}).bindTooltip(`1 · ${result.best.label}`,{permanent:true,direction:'top',className:'inspection-station-label'}).addTo(inspectionMarkers);
    const inspectionPoints=[location,stationPoint];
    if(result.runner_up){
      const runnerPoint=L.latLng(result.runner_up.lat,result.runner_up.lon);
      inspectionPoints.push(runnerPoint);
      L.polyline([location,runnerPoint],{pane:'stations',color:'#7851a9',weight:3,dashArray:'3 7',interactive:false}).addTo(inspectionMarkers);
      L.circleMarker(runnerPoint,{pane:'stations',radius:10,color:'#7851a9',weight:3,fillColor:'#fff',fillOpacity:1,interactive:false}).bindTooltip(`2 · ${result.runner_up.label}`,{permanent:true,direction:'bottom',className:'inspection-station-label inspection-runner-label'}).addTo(inspectionMarkers);
    }
    if(result.third){
      const thirdPoint=L.latLng(result.third.lat,result.third.lon);
      inspectionPoints.push(thirdPoint);
      L.polyline([location,thirdPoint],{pane:'stations',color:'#16845c',weight:3,dashArray:'10 4 2 4',interactive:false}).addTo(inspectionMarkers);
      L.circleMarker(thirdPoint,{pane:'stations',radius:10,color:'#16845c',weight:3,fillColor:'#fff',fillOpacity:1,interactive:false}).bindTooltip(`3 · ${result.third.label}`,{permanent:true,direction:'right',className:'inspection-station-label inspection-third-label'}).addTo(inspectionMarkers);
    }
    const showConnection=()=>{
      const top=Math.min(map.getSize().y*.6,(popup.getElement()?.offsetHeight||220)+40);
      map.fitBounds(L.latLngBounds(inspectionPoints),{paddingTopLeft:[65,top],paddingBottomRight:[65,65],maxZoom:12,animate:false});
    };
    const content=document.createElement('div');content.className='road-popup station-inspection';
    const title=document.createElement('div');title.className='inspection-drag-handle';title.textContent='Beste stasjoner her · dra for å flytte';title.title='Dra boksen dit du ønsker';title.tabIndex=0;
    const name=document.createElement('strong');name.textContent=`1 · ${result.best.label}`;
    const interval=result.metric==='P95';
    const stationValue=error=>`${interval?'±':''}${shortNumber(error)} ${result.unit}`;
    const comparisonValue=(error,increase)=>`${stationValue(error)} · ${shortNumber(increase)} ${result.unit} ${interval?'høyere avviksgrense':'mer feil'}`;
    const value=document.createElement('p');value.className='inspection-error';value.textContent=
      `${interval?'Anslått 95 %-intervall:':'Forventet feil'} ${stationValue(result.best.error)}`;
    const distance=document.createElement('p');distance.textContent=`${shortNumber(result.distance_km)} km unna`;
    content.append(title,name,value,distance);
    if(result.runner_up){
      const runner=document.createElement('div');runner.className='inspection-runner';
      const label=document.createElement('small');label.textContent='Nest beste stasjon';
      const station=document.createElement('strong');station.textContent=`2 · ${result.runner_up.label}`;
      const comparison=document.createElement('p');comparison.textContent=comparisonValue(result.runner_up.error,result.removal_increase);
      runner.append(label,station,comparison);content.append(runner);
    }
    if(result.third){
      const third=document.createElement('div');third.className='inspection-runner';
      const label=document.createElement('small');label.textContent='Tredje beste stasjon';
      const station=document.createElement('strong');station.textContent=`3 · ${result.third.label}`;
      const value=document.createElement('p');value.textContent=comparisonValue(result.third.error,result.third.error-result.best.error);
      third.append(label,station,value);content.append(third);
    }
    if(result.tied_count>1){
      const tied=document.createElement('small');tied.textContent=`${result.tied_count} stasjoner er like gode her`;content.append(tied);
    }
    const details=document.createElement('details'),summary=document.createElement('summary');summary.textContent='Sammenlign terreng';
    const table=document.createElement('table');
    const head=document.createElement('tr');
    const columns=['place','station',...(result.runner_up?['runner_up']:[]),...(result.third?['third']:[])];
    const headings=['','Her','1 · Best',...(result.runner_up?['2 · Nest best']:[]),...(result.third?['3 · Tredje best']:[])];
    for(const text of headings){const cell=document.createElement('th');cell.scope='col';cell.textContent=text;head.append(cell);}
    table.append(head);
    for(const item of result.terrain.filter(item=>Object.hasOwn(terrainLabels,item.field))){
      const row=document.createElement('tr'),label=document.createElement('th');label.scope='row';label.textContent=terrainLabels[item.field];row.append(label);
      for(const field of columns){const cell=document.createElement('td');cell.textContent=item[field]===null?'—':`${shortNumber(item[field])} ${item.unit}`;row.append(cell);}
      table.append(row);
    }
    details.append(summary,table);content.append(details);
    const keepVisible=()=>{
      const box=popup.getElement().getBoundingClientRect(),bounds=map.getContainer().getBoundingClientRect();
      const dx=Math.max(bounds.left+8-box.left,Math.min(0,bounds.right-8-box.right));
      const dy=Math.max(bounds.top+8-box.top,Math.min(0,bounds.bottom-8-box.bottom));
      if(dx||dy)popup.setLatLng(map.layerPointToLatLng(map.latLngToLayerPoint(popup.getLatLng()).add([dx,dy])));
    };
    details.ontoggle=()=>{popup.update();keepVisible();};
    popup.setContent(content);showConnection();
    popup.options.autoPan=false;
    const container=popup.getElement(),draggable=new L.Draggable(container,title);
    draggable.on('dragstart',()=>container.classList.add('inspection-dragging'));
    draggable.on('dragend',()=>{
      container.classList.remove('inspection-dragging');
      popup.setLatLng(map.layerPointToLatLng(L.DomUtil.getPosition(container)));
      keepVisible();
    });
    draggable.enable();
    popup.once('remove',()=>draggable.disable());
    title.onkeydown=event=>{
      const directions={ArrowLeft:[-20,0],ArrowRight:[20,0],ArrowUp:[0,-20],ArrowDown:[0,20]};
      const delta=directions[event.key];if(!delta)return;
      event.preventDefault();event.stopPropagation();
      popup.setLatLng(map.layerPointToLatLng(map.latLngToLayerPoint(popup.getLatLng()).add(delta)));
      keepVisible();
    };
  }catch(failure){
    if(serial===inspectionSerial&&revision===state.revision){map.closePopup(popup);error(failure.message);}
  }
}

async function inspectTerrain(location){
  if(!state||busy)return;
  const spacing=state.cell_m;
  try{
    const response=await fetch(`/api/terrain/point?lon=${encodeURIComponent(location.lng)}&lat=${encodeURIComponent(location.lat)}`);
    const result=await response.json();if(!response.ok)throw new Error(result.error||'Terrengverdiene kunne ikke hentes');
    if(state.cell_m!==spacing)return;
    const popup=document.createElement('div');popup.className='road-popup terrain-popup';
    const title=document.createElement('strong');title.textContent='Terrengverdier';
    const values=document.createElement('dl');
    for(const value of result.values.filter(value=>Object.hasOwn(terrainLabels,value.field))){const name=document.createElement('dt'),data=document.createElement('dd');name.textContent=terrainLabels[value.field];data.textContent=value.value===null?'Mangler':`${number(value.value,value.unit==='km'?2:1)} ${value.unit}`;
      if(mode===`terrain:${value.field}`){name.className=data.className='active-terrain';}values.append(name,data);
    }
    const add=document.createElement('button');add.innerHTML='<i data-lucide="map-pin-plus"></i>';add.append(document.createTextNode('Foreslå stasjon her'));add.onclick=()=>{map.closePopup();requestState({action:'add',lon:location.lng,lat:location.lat});};
    popup.append(title,values,add);L.popup({maxWidth:340}).setLatLng(location).setContent(popup).openOn(map);icons();
  }catch(failure){error(failure.message);}
}