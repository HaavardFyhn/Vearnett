'use strict';
const el=id=>document.getElementById(id),num=v=>Number(v).toLocaleString('en-GB',{maximumFractionDigits:3});
const map=L.map('map').setView([65,13],5),bounds=[[57.5,3],[71.5,32]];
map.fitBounds(bounds,{padding:[10,10]});
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
  {maxZoom:19,attribution:'© OpenStreetMap contributors'}).addTo(map);
map.createPane('predictionPane');map.getPane('predictionPane').style.zIndex=250;
const tiles=L.tileLayer('',{pane:'predictionPane',className:'native-tile',maxZoom:19,updateWhenIdle:true});
const markers=L.layerGroup().addTo(map);
let reference=null,result=null,busy=false,version=0,referenceVersion=0;
const resize=new ResizeObserver(()=>map.invalidateSize({pan:false}));resize.observe(el('map'));
const legend=L.control({position:'bottomright'});
legend.onAdd=()=>L.DomUtil.create('div','legend');legend.addTo(map);
async function json(url,method='GET'){
  const response=await fetch(url,{method}),data=await response.json();
  if(!response.ok)throw new Error(data.error||`HTTP ${response.status}`);
  return data;
}
function parameters(){
  return new URLSearchParams({lon:reference.lon,lat:reference.lat,day:el('day').value,variable:el('variable').value});
}
function clearPrediction(){
  version++;result=null;map.removeLayer(tiles);
  el('export').setAttribute('aria-disabled','true');el('export').removeAttribute('href');
  el('summary').textContent='';el('detail').textContent='Uncheck placement, then click a predicted cell.';
}
function showReference(){
  markers.clearLayers();
  L.circleMarker([reference.lat,reference.lon],{radius:8,color:'#000',fillColor:'#ffcc00',
    weight:2,fillOpacity:1,interactive:false}).addTo(markers);
  el('reference').textContent=`Virtual cell ${reference.cell} · ${reference.day} · ${reference.variable===0?'tg':'tn'} · archived ${num(reference.archive_c)} °C`;
}
async function chooseReference(latlng,recompute=true){
  if(busy)return;
  const id=++referenceVersion;clearPrediction();el('error').textContent='';
  el('status').textContent='Reading historical reference cell...';el('predict').disabled=true;
  try{
    const q=new URLSearchParams({lon:latlng.lng,lat:latlng.lat,day:el('day').value,variable:el('variable').value});
    const data=await json('/api/senorge-transfer/reference?'+q);
    if(id!==referenceVersion)return;
    reference=data;showReference();
    if(!el('manual').checked)el('temperature').value=String(data.archive_c);
    el('predict').disabled=false;
    if(recompute)await predict();
  }catch(error){
    if(id===referenceVersion){reference=null;markers.clearLayers();el('reference').textContent='No valid reference selected.';
      el('error').textContent=error.message;el('status').textContent='Reference selection failed';}
  }
}
async function predict(){
  if(!reference||busy)return;
  clearPrediction();busy=true;const id=version;el('error').textContent='';
  for(const name of ['predict','day','variable','manual','temperature'])el(name).disabled=true;
  el('status').textContent='Predicting temperature, then corrected RMSE from the new seNorge GPU weights...';
  try{
    const q=parameters();
    if(el('manual').checked){
      const value=el('temperature').valueAsNumber;
      if(!Number.isFinite(value))throw new Error('Enter a finite reference temperature');
      q.set('temperature',value);
    }
    const data=await json('/api/senorge-transfer/prepare?'+q,'POST');
    if(id!==version)return;
    result=data;draw();
    el('status').textContent='New seNorge prediction ready';
    el('summary').textContent=`${data.cells.toLocaleString()} native 1 km cells · reference ${num(data.reference.source_c)} °C${data.reference.manual_override?' (manual override)':' (archive)'} · median predicted corrected RMSE ${num(data.median_predicted_rmse_c)} °C. Models: ${data.selected.mean} + ${data.selected.error}.`;
    el('export').href='/api/senorge-transfer/export?key='+data.key;el('export').setAttribute('aria-disabled','false');
  }catch(error){el('error').textContent=error.message;el('status').textContent='Prediction failed';}
  finally{
    busy=false;
    for(const name of ['predict','day','variable','manual'])el(name).disabled=false;
    el('temperature').disabled=!el('manual').checked;
  }
}
function draw(){
  const metric=el('metric').value;
  legend.getContainer().textContent=metric==='rmse_c'?'Corrected RMSE °C: pale 0 → dark 5+':
    'Temperature °C: blue −20 → white 0 → red +20';
  if(!result)return;
  tiles.setUrl('/api/senorge-transfer/tile/{z}/{x}/{y}.png?'+
    new URLSearchParams({key:result.key,metric}));
  if(!map.hasLayer(tiles))tiles.addTo(map);
}
map.on('click',async event=>{
  if(busy||el('day').disabled)return;
  if(el('place').checked){await chooseReference(event.latlng);return;}
  if(!result){el('error').textContent='Place a virtual reference first.';return;}
  const id=version;
  try{
    const data=await json('/api/senorge-transfer/cell?'+new URLSearchParams({
      key:result.key,lon:event.latlng.lng,lat:event.latlng.lat}));
    if(id!==version)return;
    el('error').textContent='';
    el('detail').textContent=`Cell ${data.cell}: predicted ${num(data.temperature_c)} °C; corrected predicted RMSE ${num(data.predicted_rmse_c)} °C. `+
      (data.archived_target_c===null?'No archived target value.':
        `Archived target ${num(data.archived_target_c)} °C; signed error on this date ${num(data.signed_error_c)} °C (one error, not measured RMSE). `)+
      (data.identity_rule?'Known reference cell: explicit identity rule.':'Target archive is shown only for comparison, not used by the prediction.');
  }catch(error){if(id===version)el('error').textContent=error.message;}
});
for(const name of ['day','variable'])el(name).addEventListener('change',()=>{
  if(reference)chooseReference(L.latLng(reference.lat,reference.lon));else clearPrediction();
});
el('manual').addEventListener('change',()=>{
  clearPrediction();el('temperature').disabled=!el('manual').checked;
  if(reference&&!el('manual').checked){el('temperature').value=String(reference.archive_c);predict();}
});
el('temperature').addEventListener('input',clearPrediction);
el('predict').addEventListener('click',predict);el('metric').addEventListener('change',draw);
el('norway').addEventListener('click',()=>map.fitBounds(bounds,{padding:[10,10]}));
tiles.on('tileerror',()=>{el('error').textContent='Prediction tile failed to load. Recompute the reference or retry.';});
json('/api/senorge-transfer/options').then(data=>{
  el('warning').textContent=data.warning;
  el('day').replaceChildren(...data.days.map(day=>new Option(day,day)));
  el('day').value=data.days.includes('2024-12-15')?'2024-12-15':data.days[data.days.length-1];
  el('day').disabled=false;el('variable').disabled=false;
  el('status').textContent='New seNorge models verified. Click any supported land cell.';
  draw();
}).catch(error=>{el('error').textContent=error.message;el('status').textContent='Model/archive verification failed';});
