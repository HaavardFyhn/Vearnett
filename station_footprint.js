'use strict';
const element=id=>document.getElementById(id);
const map=L.map('map').setView([61.4,6.35],9);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',{
  maxZoom:19,attribution:'© OpenStreetMap contributors'
}).addTo(map);
const stationsLayer=L.layerGroup().addTo(map);
let catalogue=[],state=null,overlay=null,referenceMarker=null,busy=false,selection=null;
let networkTimer=null,pointRequest=0;
const number=value=>Number(value).toLocaleString('nb-NO',{maximumFractionDigits:3});

async function json(url){
  const response=await fetch(url);
  const result=await response.json();
  if(!response.ok)throw new Error(result.error||`HTTP ${response.status}`);
  return result;
}
function setBusy(value){
  busy=value;
  element('station').disabled=value;
  element('target').disabled=value;
  element('station-search').disabled=value;
  element('view-mode').disabled=value;
  element('click-mode').disabled=value||element('view-mode').value!=='single';
  element('export').setAttribute('aria-disabled',String(value||!state));
}
function filterStations(selectedIdentifier){
  const search=element('station-search').value.trim().toLowerCase();
  const selected=typeof selectedIdentifier==='string'?selectedIdentifier:element('station').value;
  const matches=catalogue.filter(item=>`${item.station_id} ${item.name}`.toLowerCase().includes(search));
  const options=matches.slice(0,50);
  const active=matches.find(item=>item.station_id===selected);
  if(active&&!options.includes(active))options.push(active);
  element('station').replaceChildren(new Option('Velg en stasjon …',''));
  for(const station of options){
    const option=new Option(`${station.station_id} · ${station.name}${station.supported?'':' · utenfor modellruter'}`,station.station_id);
    option.disabled=!station.supported;
    element('station').add(option);
  }
  if(options.some(item=>item.station_id===selected))element('station').value=selected;
}
function requestQuery(reference){
  return new URLSearchParams({...reference,target:element('target').value,cell_m:1000});
}
function networkQuery(){
  const query=new URLSearchParams({target:element('target').value,mode:element('view-mode').value});
  if(element('station').value)query.set('station',element('station').value);
  return query;
}
function legend(mode){
  const lost=mode==='lost',reduction=mode==='reduction';
  element('legend-label').textContent=lost?'Dekningstap uten valgt stasjon':reduction?'Valgt stasjons RMSE-reduksjon · °C':'Anslått seNorge-representasjonsfeil · RMSE °C';
  element('legend-ticks').replaceChildren(...(lost?['0 · fortsatt dekning','1 · mister dekning']:['0','1','2','≥ 3 °C']).map(text=>{
    const span=document.createElement('span');span.textContent=text;return span;
  }));
}
async function networkReady(){
  const status=await json('/api/footprint/network/status');
  if(status.state==='ready')return true;
  if(status.state==='failed')throw new Error(`Nettberegningen feilet: ${status.error}. Se footprint_network_preparation.log og start beregningen på nytt.`);
  if(status.state==='missing'){
    const response=await fetch('/api/footprint/network/prepare',{method:'POST'});
    if(!response.ok){const error=await response.json();throw new Error(error.error||`HTTP ${response.status}`);}
  }
  element('status').textContent=`Beregner Frost-nettet: ${status.completed||0}/${status.stations} stasjoner · alle seks mål. Enkeltkart er fortsatt tilgjengelig.`;
  networkTimer=setTimeout(()=>{networkTimer=null;if(element('view-mode').value!=='single')show(selection||{},false);},3000);
  return false;
}
async function show(reference,fit=true){
  if(busy)return;
  clearTimeout(networkTimer);networkTimer=null;
  const mode=element('view-mode').value,network=mode!=='single';
  if(network&&mode!=='network'&&!element('station').value){
    element('error').textContent='Velg en Frost-stasjon for å vise bidrag eller dekningstap.';
    return;
  }
  setBusy(true);
  pointRequest++;
  element('error').textContent='';
  element('point').textContent='';
  element('ranking').replaceChildren();
  element('status').textContent=network?'Laster Frost-nettkart …':'Beregner seks footprint-kart på CPU …';
  try{
    if(network&&!await networkReady())return;
    const query=network?networkQuery():requestQuery(reference);
    const base=network?'/api/footprint/network':'/api/footprint';
    const result=await json(`${base}?${query}`);
    if(!network){query.set('reference',result.reference);query.delete('lon');query.delete('lat');query.delete('station');}
    const response=await fetch(`${base}.png?${query}`);
    if(!response.ok){
      const error=await response.json();
      throw new Error(error.error||'Footprint-bildet kunne ikke lastes');
    }
    const url=URL.createObjectURL(await response.blob());
    const image=await new Promise((resolve,reject)=>{
      const value=new Image();
      value.onload=()=>resolve(value);
      value.onerror=()=>{URL.revokeObjectURL(url);reject(new Error('Footprint-bildet kunne ikke leses'));};
      value.src=url;
    });
    if(overlay){URL.revokeObjectURL(overlay.getElement().src);map.removeLayer(overlay);}
    overlay=L.imageOverlay(image.src,result.bounds,{opacity:Number(element('opacity').value),interactive:false}).addTo(map);
    if(referenceMarker)map.removeLayer(referenceMarker);
    referenceMarker=null;
    const station=network?catalogue.find(item=>item.station_id===element('station').value):result;
    if(station)referenceMarker=L.circleMarker([station.lat,station.lon],{radius:8,color:'#fff',weight:2,fillColor:'#146b4b',fillOpacity:1})
      .bindTooltip(network?`Valgt Frost-stasjon: ${station.station_id}`:'Referansens 1 km-rutesenter').addTo(map);
    if(fit){
      if(network&&mode!=='network'&&station)map.setView([station.lat,station.lon],9);
      else map.fitBounds(result.bounds,{padding:[15,15]});
    }
    state={...result,mode};if(!network)selection={reference:result.reference};
    element('export').href=`${base}/export?${query}`;
    legend(mode);
    if(network){
      let details=`${result.label} · ${result.stations} Frost-stasjoner\nDekker ${number(result.covered_cells)} km² land · ${number(result.uncovered_cells)} landruter uten dekning\nKlikk en rute for å rangere stasjonene som dekker stedet.`;
      if(mode!=='network')details+=`\n${result.station}: mister ${number(result.lost_coverage_area_km2)} km² dekning ved fjerning.`;
      if(mode==='reduction')details+=`\nPositivt numerisk bidrag: ${number(result.affected_area_km2)} km².\nDekningstap er NoData i bidragskartet; se eget dekningstap-kart.`;
      if(mode!=='lost'){
        const mean=result.mean_c===null?'ingen':result.mean_c>0&&result.mean_c<.001?'< 0,001 °C':number(result.mean_c)+' °C';
        details+=`\nGjennomsnitt over numerisk støttede celler: ${mean}.`;
      }
      element('details').textContent=details;
    }else element('details').textContent=`${result.label} · ${result.valid_cells} støttede 1 km-celler\nReferanserute: ${number(result.x)}, ${number(result.y)} (EPSG:25833)\nHøyde: ${number(result.elevation_m)} m · hav: ${number(result.coast_ocean_distance_m/1000)} km · fjord: ${number(result.coast_fjord_distance_m/1000)} km\nKartets gjennomsnittlige anslag: ${number(result.mean_c)} °C`;
    element('scores').textContent=`Romlig test (ikke denne stasjonen):\nCNN RMSE: ${number(result.test.rmse)} °C · bias: ${number(result.test.bias)} °C\nXGBoost RMSE: ${number(result.baseline_test.rmse)} °C`;
    element('warning').textContent=result.warning;
    element('status').textContent=`Klar · 1 000 m · ${result.selected}`;
    document.title=`Værnett | ${result.label} · Footprint-CNN · 1 km`;
  }catch(error){
    if(state){element('target').value=state.target;element('view-mode').value=state.mode;}
    element('error').textContent=error.message;
    element('status').textContent=state?'Forrige kart er fortsatt vist':'Ingen footprint-prediksjon';
  }finally{setBusy(false);}
}
element('station-search').addEventListener('input',filterStations);
element('station').onchange=()=>{
  if(element('station').value)show({station:element('station').value});
};
element('target').onchange=()=>{if(selection)show(selection,false);};
element('view-mode').onchange=()=>{
  const mode=element('view-mode').value;
  if(mode!=='single')element('click-mode').value='inspect';
  const fit=!state||state.mode==='single'||mode==='single'||mode==='network'||state.mode==='network';
  show(element('station').value?{station:element('station').value}:selection||{},fit);
};
element('opacity').oninput=()=>{if(overlay)overlay.setOpacity(Number(element('opacity').value));};
map.on('click',async event=>{
  if(busy)return;
  if(element('view-mode').value==='single'&&element('click-mode').value==='reference'){
    element('station').value='';
    await show({lon:event.latlng.lng,lat:event.latlng.lat});
  }else if(state){
    const token=++pointRequest;
    try{
      const network=state.mode!=='single';
      const query=new URLSearchParams({target:state.target,point_lon:event.latlng.lng,point_lat:event.latlng.lat});
      if(network){if(element('station').value)query.set('station',element('station').value);}
      else query.set('reference',state.reference);
      const point=await json(`/api/footprint/${network?'network/':''}point?${query}`);
      if(token!==pointRequest)return;
      element('ranking').replaceChildren();element('error').textContent='';
      element('point').textContent=point.supported?`1 km-rute (${point.x}, ${point.y}): ${number(point.value_c)} °C RMSE${network?`\nBeste stasjon: ${point.best_station} · ${point.coverage_count} stasjoner dekker ruten`:''}`:point.reason;
      if(network&&point.supported){
        const options=point.ranking.slice(0,30);
        const selected=point.ranking.find(item=>item.station_id===element('station').value);
        if(selected&&!options.includes(selected))options.push(selected);
        if(element('station').value&&!point.selected_station)element('point').textContent+='\nValgt stasjon dekker ikke denne ruten.';
        for(const item of options){
          const row=document.createElement('li'),button=document.createElement('button');
          button.textContent=`${item.station_id} · ${item.name}\nRMSE ${number(item.value_c)} °C · ${item.loses_coverage?'eneste dekning':`bidrag ${number(item.removal_reduction_c)} °C`}`;
          button.onclick=()=>{
            element('station-search').value='';filterStations(item.station_id);
            element('station').value=item.station_id;element('view-mode').value='reduction';
            show({station:item.station_id},false);
          };
          row.append(button);element('ranking').append(row);
        }
      }
    }catch(error){if(token===pointRequest)element('error').textContent=error.message;}
  }
});
(async()=>{
  try{
    const options=await json('/api/footprint/options');
    element('scenario-link').hidden=options.standalone;
    catalogue=options.stations;
    for(const target of options.targets)element('target').add(new Option(target.label,target.id));
    element('target').value='winter_tn';
    element('warning').textContent=options.warning;
    filterStations();
    for(const station of catalogue){
      const marker=L.circleMarker([station.lat,station.lon],{radius:4,color:'#45524e',weight:1,fillOpacity:.8});
      const label=document.createElement('span');label.textContent=`${station.station_id} · ${station.name}`;
      marker.bindTooltip(label).addTo(stationsLayer);
      marker.on('click',event=>{
        L.DomEvent.stopPropagation(event.originalEvent);
        if(busy)return;
        element('station-search').value='';filterStations(station.station_id);
        element('station').value=station.station_id;
        show({station:station.station_id});
      });
    }
    setBusy(false);
    const initial=catalogue.find(item=>item.station_id==='SN56960'&&item.supported)||catalogue.find(item=>item.supported);
    if(initial){filterStations(initial.station_id);element('station').value=initial.station_id;await show({station:initial.station_id});}
    else element('status').textContent='Ingen Frost-stasjon finnes i modellens landruter; klikk en referanse på kartet';
  }catch(error){element('error').textContent=error.message;element('status').textContent='Modellen kunne ikke lastes';}
})();
