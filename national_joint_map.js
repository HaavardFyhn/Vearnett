'use strict';
const el=id=>document.getElementById(id),number=v=>Number(v).toLocaleString('en-GB',{maximumFractionDigits:2});
const map=L.map('map',{preferCanvas:true}).setView([65,13],5);
const nationalBounds=[[57.5,3],[71.5,32]];
map.fitBounds(nationalBounds,{padding:[10,10]});
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',{maxZoom:19,attribution:'© OpenStreetMap contributors'}).addTo(map);
const sources=L.layerGroup().addTo(map);
let ready=false,running=false,generation=0,detailVersion=0,pollTimer=null;
let nationalView=true;
let preparedDay=null,preparedHour=null,removalOptionsKey=null;
const removedMarker=L.layerGroup().addTo(map);
map.createPane('networkStationPane');map.getPane('networkStationPane').style.zIndex=650;
const networkRenderer=L.svg({pane:'networkStationPane'});
const stationNetwork=L.layerGroup().addTo(map);
map.createPane('predictionPane');map.getPane('predictionPane').style.zIndex=250;
const nativeTiles=L.tileLayer('',{pane:'predictionPane',className:'native-prediction-tile',
  tileSize:256,maxZoom:19,opacity:1,updateWhenIdle:true,keepBuffer:1});
let nativeTileUrl=null;
function drawStationNetwork(catalogue){
  stationNetwork.clearLayers();
  for(const station of catalogue){
    const marker=L.circleMarker([station.lat,station.lon],{
      renderer:networkRenderer,radius:station.available?5:4,color:station.available?'#00441b':'#555',
      fillColor:station.available?'#31a354':'#bbb',fillOpacity:1,weight:1,bubblingMouseEvents:false
    }).addTo(stationNetwork);
    marker.getElement().setAttribute('data-station-id',station.station_id);
    marker.getElement().setAttribute('aria-label',`${station.name}: ${station.available?'click to remove reading':'no QC reading'}`);
    marker.getElement().setAttribute('tabindex','0');
    marker.bindTooltip(text('span',`${station.station_id} · ${station.name} · `+
      (station.available?`${number(station.source_c)} °C · click to remove`:'no QC reading at this time')));
    marker.on('click',()=>{
      if(!ready||running){el('station-selection').textContent='Wait until the national prediction is ready.';return;}
      const message=station.available?
        `Removing ${station.name} (${station.station_id}), reading ${number(station.source_c)} °C. Displaying its effect across the map.`:
        `${station.name} (${station.station_id}) has no selected-sensor QC reading at ${preparedDay} ${String(preparedHour).padStart(2,'0')} UTC. It was not used, so removing it has no effect. Full network restored.`;
      el('removed-station').value=station.available?station.station_id:'';
      el('scenario').value='change';el('metric').value='radius_c';
      removalChanged();el('station-selection').textContent=message;
      marker.openTooltip();
    });
    marker.getElement().addEventListener('keydown',event=>{
      if(event.key==='Enter'||event.key===' '){event.preventDefault();marker.fire('click');}
    });
  }
}
el('show-stations').addEventListener('change',()=>{
  if(el('show-stations').checked)stationNetwork.addTo(map);else map.removeLayer(stationNetwork);
});
el('restore-network').addEventListener('click',()=>{
  el('removed-station').value='';removalChanged();el('station-selection').textContent='All available station readings restored.';
});
function removalQuery(){
  return new URLSearchParams({day:preparedDay,hour:preparedHour,
    station:el('removed-station').value,mode:el('scenario').value});
}
const resizeObserver=new ResizeObserver(()=>{
  map.invalidateSize({pan:false});
});
resizeObserver.observe(el('map'));
map.on('dragstart',()=>{nationalView=false;});
el('map').addEventListener('wheel',()=>{nationalView=false;},{passive:true});
async function json(url,method='GET'){
  const r=await fetch(url,{method}),data=await r.json();if(!r.ok)throw new Error(data.error||`HTTP ${r.status}`);return data;
}
function text(tag,value){const node=document.createElement(tag);node.textContent=value;return node;}
const legend=L.control({position:'bottomright'});legend.onAdd=()=>L.DomUtil.create('div','legend');legend.addTo(map);
function draw(){
  const metric=el('metric').value;
  legend.getContainer().textContent=metric==='radius_c'?'Radius ± °C: pale0 → dark25+':
    metric==='temperature_c'?'Temperature °C: blue−20 → white0 → red+20':'Source count: pale0 → dark6';
  if(el('removed-station').value&&el('scenario').value==='change')legend.getContainer().textContent=
    'CHANGE without − full (°C): blue negative, red positive, pale zero. Scale −2 to +2; grey = lost prediction.';
  if(!ready)return;
  const q=removalQuery();q.set('metric',metric);
  if(!el('removed-station').value)q.set('mode','full');
  const url='/api/national-removal/tile/{z}/{x}/{y}.png?'+q;
  if(url!==nativeTileUrl){nativeTileUrl=url;nativeTiles.setUrl(url);}
  if(!map.hasLayer(nativeTiles))nativeTiles.addTo(map);
}
async function loadGrid(){
  if(!ready)return;
  const id=++generation;
  el('error').textContent='';
  draw();
  try{
    const data=await json('/api/national-removal/summary?'+removalQuery());
    if(id!==generation)return;el('resolution').textContent=
      'Native 1 km raster predictions at every zoom (nearest-cell rendering, no 20 km aggregation). Click any supported land location to inspect its exact 1 km cell; the map will not move.';
    removedMarker.clearLayers();
    const r=data.removal;
    el('removal-summary').textContent=r?
      `${r.station} ${r.name}: ${r.affected_cells.toLocaleString()} cells use this reading; ${r.unchanged_cells.toLocaleString()} cells unchanged; ${r.lost_cells.toLocaleString()} cells become NoData. Radius increases in ${r.positive_radius_cells.toLocaleString()} cells and decreases in ${r.negative_radius_cells.toLocaleString()}. ${r.explanation}`:'';
    if(r){
      L.circleMarker([r.lat,r.lon],{pane:'networkStationPane',radius:10,color:'#000',weight:3,
        fillColor:'#ffcc00',fillOpacity:1,interactive:false})
        .bindTooltip(text('span',`REMOVED ${r.station} ${r.name}`),{permanent:true}).addTo(removedMarker);
    }
  }
  catch(error){if(id===generation)el('error').textContent=error.message;}
}
function setStatus(data){
  el('status').textContent=data.detail||data.state;
  if(data.state==='ready'){
    ready=true;running=false;clearTimeout(pollTimer);
    const m=data.metadata;
    preparedDay=m.day;preparedHour=m.hour;
    el('day').value=m.day;el('hour').value=String(m.hour);
    loadRemovalOptions();
    el('status').textContent=`National prediction ready · ${m.time}`;
    el('summary').textContent=`${m.cells.toLocaleString()} predicted 1 km land cells · ${m.no_prediction_cells.toLocaleString()} atlas cells NoData · ${m.sources} observed sources (${m.nontraining_sources} outside training network) · median radius ±${number(m.interval_radius_median_c)} °C.`;
    el('export').setAttribute('aria-disabled','false');el('export').href='/api/national-joint/export';
    el('prepare').disabled=false;el('day').disabled=false;el('hour').disabled=false;
    if(nationalView){map.invalidateSize({pan:false});map.fitBounds(nationalBounds,{padding:[10,10],animate:false});}
    loadGrid();return;
  }
  if(data.state==='failed'){
    running=false;ready=false;el('error').textContent=data.error;el('prepare').disabled=false;el('day').disabled=false;el('hour').disabled=false;return;
  }
  if(data.state==='running'){running=true;pollTimer=setTimeout(poll,2000);}
}
async function poll(){try{setStatus(await json('/api/national-joint/status'));}catch(error){el('error').textContent=error.message;}}
async function prepare(){
  if(running)return;ready=false;sources.clearLayers();detailVersion++;generation++;
  el('error').textContent='';el('export').setAttribute('aria-disabled','true');el('export').removeAttribute('href');
  el('removed-station').disabled=true;el('scenario').disabled=true;removedMarker.clearLayers();
  map.removeLayer(nativeTiles);nativeTileUrl=null;
  stationNetwork.clearLayers();el('station-selection').textContent='Updating station availability for the new prediction.';
  for(const id of ['prepare','day','hour'])el(id).disabled=true;
  try{setStatus(await json('/api/national-joint/prepare?'+new URLSearchParams({day:el('day').value,hour:el('hour').value}),'POST'));}
  catch(error){el('error').textContent=error.message;el('prepare').disabled=false;el('day').disabled=false;el('hour').disabled=false;}
}
async function detail(p,center,nativeCell=null){
  const id=++detailVersion;sources.clearLayers();el('detail').textContent='Loading joint source-removal predictions...';
  try{
    const data=nativeCell||await json('/api/national-joint/detail?cell='+p.cell);if(id!==detailVersion)return;
    const container=el('detail');container.replaceChildren(text('strong',`Cell ${data.cell}`),
      text('p',`${number(data.temperature_c)} °C · radius ±${number(data.radius_c)} °C · [${number(data.lower_c)}, ${number(data.upper_c)}] °C`),
      text('p','Δ is without minus full. Positive radius change means wider without that reading; negative narrower. Green/purple mark positive/negative changes. Not causal or additive station value.'));
    const selected=data.sources.find(s=>s.station_id===el('removed-station').value);
    if(el('removed-station').value){
      container.append(text('p',!selected?'Selected removed reading was not used here: no change.':
        !selected.deletion_prediction_available?'Removing the selected station gives NoData: it was the only source.':
        `WITHOUT ${selected.station_id}: ${number(data.temperature_c+selected.temperature_change_c)} °C · radius ±${number(data.radius_c+selected.radius_change_c)} °C. Changes: temperature ${number(selected.temperature_change_c)} °C; radius ${number(selected.radius_change_c)} °C.`));
    }
    const table=document.createElement('table'),head=document.createElement('tr');
    for(const name of ['Source','Reading °C','Δ radius °C','Δ temperature °C'])head.append(text('th',name));table.append(head);
    for(const s of data.sources){
      const row=document.createElement('tr');
      for(const value of [`${s.station_id} ${s.name}${s.extrapolation?' (extrapolation)':''}`,number(s.source_c),
        s.deletion_prediction_available?number(s.radius_change_c):'NoData',
        s.deletion_prediction_available?number(s.temperature_change_c):'NoData'])row.append(text('td',value));
      table.append(row);
      const removeButton=text('button','Remove across map');
      removeButton.style.fontSize='11px';removeButton.addEventListener('click',event=>{
        event.stopPropagation();el('removed-station').value=s.station_id;el('scenario').value='change';removalChanged();
      });
      const action=document.createElement('td');action.append(removeButton);row.append(action);
      const marker=L.circleMarker([s.lat,s.lon],{radius:7,color:'#111',fillOpacity:1,bubblingMouseEvents:false,
        fillColor:s.radius_change_c===null?'#2171b5':s.radius_change_c>=0?'#238b45':'#756bb1'}).addTo(sources);
      marker.on('click',()=>{
        el('removed-station').value=s.station_id;el('scenario').value='change';removalChanged();
      });
      marker.bindTooltip(text('span',`${s.station_id} ${s.name} · ${number(s.source_c)} °C`));
      L.polyline([center,[s.lat,s.lon]],{color:'#555',weight:1,dashArray:'4 4'}).addTo(sources);
      row.tabIndex=0;row.style.cursor='pointer';
      const focus=()=>{map.panTo([s.lat,s.lon]);marker.openTooltip();};
      row.addEventListener('click',focus);row.addEventListener('keydown',event=>{if(event.key==='Enter')focus();});
    }
    container.append(table);
    if(data.sources.length===1)container.append(text('p','Removing the sole source gives NoData, not zero uncertainty.'));
  }catch(error){if(id===detailVersion){el('error').textContent=error.message;el('detail').textContent='Source diagnostic failed';}}
}
async function loadRemovalOptions(){
  const key=`${preparedDay}|${preparedHour}`;
  if(removalOptionsKey===key){el('removed-station').disabled=false;return;}
  const old=el('removed-station').value;
  try{
    const data=await json('/api/national-removal/options?'+new URLSearchParams({day:preparedDay,hour:preparedHour}));
    if(key!==`${preparedDay}|${preparedHour}`)return;
    el('removed-station').replaceChildren(new Option('Full network (no removal)',''),
      ...data.stations.map(s=>new Option(`${s.station_id} · ${s.name} · ${number(s.source_c)} °C`,s.station_id)));
    if(data.stations.some(s=>s.station_id===old))el('removed-station').value=old;
    drawStationNetwork(data.catalogue);
    removalOptionsKey=key;el('removed-station').disabled=false;removalChanged();
  }catch(error){el('error').textContent=error.message;}
}
function removalChanged(){
  const selected=!!el('removed-station').value;
  el('station-selection').textContent=selected?
    `Selected reading removed: ${el('removed-station').selectedOptions[0].textContent}`:'Full network: no reading removed.';
  el('scenario').disabled=!selected;
  if(selected&&el('metric').value==='source_count')el('metric').value='radius_c';
  el('metric').options[2].disabled=selected&&el('scenario').value==='change';
  el('removal-export').hidden=!selected;
  if(selected)el('removal-export').href='/api/national-removal/export?'+removalQuery();
  detailVersion++;sources.clearLayers();el('detail').textContent='Zoom and click a cell to compare full versus without-station predictions.';
  loadGrid();
}
el('removed-station').addEventListener('change',removalChanged);
el('scenario').addEventListener('change',removalChanged);
nativeTiles.on('tileerror',()=>{el('error').textContent='A native prediction tile failed to load. Retry or check prediction status.';});
map.on('click',async event=>{
  if(!ready||running)return;
  const id=++detailVersion;
  try{
    const q=new URLSearchParams({lat:event.latlng.lat,lon:event.latlng.lng,day:preparedDay,hour:preparedHour});
    const cell=await json('/api/national-removal/cell?'+q);
    if(id!==detailVersion)return;
    el('error').textContent='';detail({cell:cell.cell},event.latlng,cell);
  }catch(error){if(id===detailVersion){el('detail').textContent=error.message;el('error').textContent=error.message;}}
});
el('prepare').addEventListener('click',prepare);el('metric').addEventListener('change',draw);
el('national').addEventListener('click',()=>{nationalView=true;map.invalidateSize({pan:false});map.fitBounds(nationalBounds,{padding:[10,10]});});
json('/api/national-joint/options').then(async options=>{
  el('warning').textContent=options.warning;
  el('day').replaceChildren(...options.days.map(value=>new Option(value,value)));el('day').value='2024-12-15';
  el('day').disabled=false;el('prepare').disabled=false;
  const state=await json('/api/national-joint/status');setStatus(state);
}).catch(error=>{el('error').textContent=error.message;});
