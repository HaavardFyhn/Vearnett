'use strict';
const el=id=>document.getElementById(id),num=v=>Number(v).toLocaleString('en-GB',{maximumFractionDigits:3});
const map=L.map('map').setView([65,13],5),bounds=[[57.5,3],[71.5,32]];
map.fitBounds(bounds,{padding:[10,10]});
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
  {maxZoom:19,attribution:'© OpenStreetMap contributors'}).addTo(map);
map.createPane('predictionPane');map.getPane('predictionPane').style.zIndex=250;
map.createPane('stationPane');map.getPane('stationPane').style.zIndex=650;
const renderer=L.svg({pane:'stationPane'}),stations=L.layerGroup().addTo(map);
const tiles=L.tileLayer('',{pane:'predictionPane',className:'native-tile',maxZoom:19,updateWhenIdle:true});
const removedMarker=L.layerGroup().addTo(map);
const resize=new ResizeObserver(()=>map.invalidateSize({pan:false}));resize.observe(el('map'));
let ready=null,catalogue=[],version=0,pollTimer=null,busy=false;
const legend=L.control({position:'bottomright'});
legend.onAdd=()=>L.DomUtil.create('div','legend');legend.addTo(map);
async function json(url,method='GET'){
  const r=await fetch(url,{method}),data=await r.json();
  if(!r.ok)throw new Error(data.error||`HTTP ${r.status}`);return data;
}
function query(){
  return new URLSearchParams({key:ready.key,station:el('station').value,
    mode:el('station').value?el('mode').value:'full',metric:'rmse_c'});
}
function draw(){
  if(!ready)return;
  const q=query();tiles.setUrl('/api/senorge-network/tile/{z}/{x}/{y}.png?'+q);
  if(!map.hasLayer(tiles))tiles.addTo(map);
  el('export').href='/api/senorge-network/export?'+q;el('export').setAttribute('aria-disabled','false');
  legend.getContainer().textContent=q.get('mode')==='change'?
    'RMSE increase °C: pale 0 → dark 5+; grey = no remaining source':
    'Corrected RMSE °C: pale 0 → dark 5+; unsupported = transparent';
}
async function changed(){
  if(!ready)return;
  const id=++version;el('error').textContent='';draw();removedMarker.clearLayers();
  const station=catalogue.find(s=>s.station_id===el('station').value);
  el('selection').textContent=station?`REMOVED ${station.name} (${station.station_id})`:'Full network: all eligible references';
  if(station)L.circleMarker([station.lat,station.lon],{pane:'stationPane',radius:10,color:'#000',
    fillColor:'#ffcc00',fillOpacity:1,weight:2,interactive:false}).addTo(removedMarker);
  el('detail').textContent='Click land for exact full/without RMSE and the next-best source.';
  try{
    const r=await json('/api/senorge-network/summary?'+query());if(id!==version)return;
    el('removal').textContent=station?
      `${r.affected_cells.toLocaleString()} cells select this station. Removal increases RMSE in ${r.increased_cells.toLocaleString()} cells; ${r.lost_cells.toLocaleString()} lose their only source (NoData); ${r.unchanged_cells.toLocaleString()} remain unchanged. Maximum finite increase ${num(r.max_increase_c)} °C.`:'';
  }catch(error){if(id===version)el('error').textContent=error.message;}
}
function showStations(){
  stations.clearLayers();
  for(const station of catalogue){
    const marker=L.circleMarker([station.lat,station.lon],{renderer,radius:station.available?5:4,
      color:station.available?'#00441b':'#555',fillColor:station.available?'#31a354':'#bbb',
      weight:1,fillOpacity:1,bubblingMouseEvents:false}).addTo(stations);
    marker.getElement().setAttribute('data-station-id',station.station_id);
    marker.getElement().setAttribute('tabindex','0');
    marker.getElement().setAttribute('aria-label',`${station.name}: remove reference`);
    const tooltip=document.createElement('span');
    tooltip.textContent=`${station.station_id} ${station.name} · `+
      (station.available?`seNorge ${num(station.source_c)} °C · click to remove`:station.reason);
    marker.bindTooltip(tooltip);
    const remove=()=>{
      if(!ready||busy)return;
      el('station').value=station.station_id;el('mode').value='change';changed();
      if(!station.available)el('selection').textContent=
        `${station.name}: ${station.reason}. This station was not eligible, so removal has no effect.`;
    };
    marker.on('click',remove);
    marker.getElement().addEventListener('keydown',event=>{
      if(event.key==='Enter'||event.key===' '){event.preventDefault();remove();}
    });
  }
}
function status(data){
  el('status').textContent=data.detail;
  if(data.state==='running'){busy=true;pollTimer=setTimeout(poll,1500);return;}
  clearTimeout(pollTimer);busy=false;
  for(const name of ['prepare','day','variable'])el(name).disabled=false;
  if(data.state==='failed'){el('error').textContent=data.detail;return;}
  if(data.state!=='ready')return;
  ready=data.metadata;catalogue=data.catalogue;
  el('day').value=ready.day;el('variable').value=String(ready.variable);
  el('station').replaceChildren(new Option('Full network',''),
    ...catalogue.map(s=>new Option(`${s.station_id} ${s.name}`,s.station_id)));
  el('station').disabled=false;el('mode').value='full';showStations();
  el('summary').textContent=`${ready.cells.toLocaleString()} predicted 1 km land cells; ${ready.no_prediction_cells.toLocaleString()} atlas cells have no reference within 46 km. ${ready.eligible_stations} eligible references of ${ready.stations} visible Frost stations. Median corrected RMSE ${num(ready.median_predicted_rmse_c)} °C.`;
  changed();
}
async function poll(){
  try{status(await json('/api/senorge-network/status'));}
  catch(error){el('error').textContent=error.message;busy=false;el('prepare').disabled=false;}
}
async function prepare(){
  if(busy)return;
  ready=null;version++;map.removeLayer(tiles);removedMarker.clearLayers();
  el('export').setAttribute('aria-disabled','true');el('error').textContent='';
  el('station').disabled=true;el('summary').textContent='';el('removal').textContent='';
  for(const name of ['prepare','day','variable'])el(name).disabled=true;
  try{status(await json('/api/senorge-network/prepare?'+
    new URLSearchParams({day:el('day').value,variable:el('variable').value}),'POST'));}
  catch(error){el('error').textContent=error.message;for(const name of ['prepare','day','variable'])el(name).disabled=false;}
}
map.on('click',async event=>{
  if(!ready||busy)return;
  const id=version;
  try{
    const q=query();q.set('lon',event.latlng.lng);q.set('lat',event.latlng.lat);
    const data=await json('/api/senorge-network/cell?'+q);if(id!==version)return;
    el('error').textContent='';
    el('detail').textContent=`Cell ${data.cell}: full corrected RMSE ${num(data.full_rmse_c)} °C; winning source ${data.source.station_id} ${data.source.name}. `+
      `Next-best: ${data.next_source?data.next_source.station_id+' '+data.next_source.name:'none'}. `+
      (data.lost_prediction?'Removing this station leaves no prediction.':
        `Without selected station: ${num(data.without_rmse_c)} °C; increase ${num(data.increase_c)} °C.`);
  }catch(error){if(id===version){el('error').textContent=error.message;el('detail').textContent=error.message;}}
});
el('prepare').addEventListener('click',prepare);el('station').addEventListener('change',changed);
for(const name of ['day','variable'])el(name).addEventListener('change',()=>{
  ready=null;version++;map.removeLayer(tiles);stations.clearLayers();removedMarker.clearLayers();
  el('station').disabled=true;el('export').setAttribute('aria-disabled','true');
  el('summary').textContent='';el('removal').textContent='';
  el('status').textContent='Date/variable changed. Calculate the network for the new selection.';
});
el('mode').addEventListener('change',changed);
el('restore').addEventListener('click',()=>{el('station').value='';el('mode').value='full';changed();});
el('show-stations').addEventListener('change',()=>{
  if(el('show-stations').checked)stations.addTo(map);else map.removeLayer(stations);
});
el('norway').addEventListener('click',()=>map.fitBounds(bounds,{padding:[10,10]}));
tiles.on('tileerror',()=>{el('error').textContent='Network tile failed to load; retry or recalculate the network.';});
json('/api/senorge-network/options').then(async data=>{
  el('warning').textContent=data.warning;
  el('day').replaceChildren(...data.days.map(day=>new Option(day,day)));
  el('day').value='2024-12-15';el('day').disabled=false;el('prepare').disabled=false;
  const current=await json('/api/senorge-network/status');
  if(current.state==='idle')prepare();else status(current);
}).catch(error=>{el('error').textContent=error.message;});
