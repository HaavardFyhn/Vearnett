'use strict';
const el=id=>document.getElementById(id);
const map=L.map('map').setView([61.4,6.4],9);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',{
  maxZoom:19,attribution:'© OpenStreetMap contributors'
}).addTo(map);
const layer=L.layerGroup().addTo(map);
let options=null,generation=0,timer=null;
const number=v=>Number(v).toLocaleString('nb-NO',{maximumFractionDigits:2});
function color(value){
  const colors=[[5,48,97],[67,147,195],[247,247,247],[214,96,77],[103,0,31]];
  const p=Math.max(0,Math.min(4,(value+20)/10)),i=Math.min(3,Math.floor(p)),f=p-i;
  return `rgb(${colors[i].map((v,k)=>Math.round(v+(colors[i+1][k]-v)*f)).join(',')})`;
}
async function json(url){
  const response=await fetch(url),payload=await response.json();
  if(!response.ok)throw new Error(payload.error||`HTTP ${response.status}`);
  return payload;
}
function draw(data){
  layer.clearLayers();el('nodes').replaceChildren();
  const stations=new Map(options.stations.map(s=>[s.station_id,s]));
  for(const node of data.nodes){
    const s=stations.get(node.station_id),parent=stations.get(node.parent);
    if(parent)L.polyline([[parent.lat,parent.lon],[s.lat,s.lon]],{
      color:color(node.temperature_c),weight:3,dashArray:node.hop>1?'5 5':null
    }).addTo(layer);
    const marker=L.circleMarker([s.lat,s.lon],{
      radius:node.hop===0?11:7,color:'#222',weight:node.hop===0?3:1,
      fillColor:color(node.temperature_c),fillOpacity:1
    }).addTo(layer);
    const tooltip=document.createElement('span');
    tooltip.textContent=`${s.station_id} · ${s.name}: ${number(node.temperature_c)} °C · syntetisk · hopp ${node.hop}`;
    marker.bindTooltip(tooltip);
    const tr=document.createElement('tr');
    for(const value of [node.station_id,node.parent||'Start',node.hop,number(node.temperature_c)]){
      const td=document.createElement('td');td.textContent=value;tr.append(td);
    }el('nodes').append(tr);
  }
  el('status').textContent=`${data.nodes.length} syntetiske stasjoner · ${data.time} · ${options.selected}`;
  el('unreachable').textContent=`Ikke nådd innen valgt hopptall/45 km per kant: ${data.unreachable.join(', ')||'Ingen'}`;
}
async function simulate(id){
  el('error').textContent='';el('status').textContent='Beregner syntetiske overføringer ...';
  layer.clearLayers();el('nodes').replaceChildren();
  const query=new URLSearchParams({source:el('source').value,time:el('time').value,
    temperature:el('temperature').value,hops:el('hops').value});
  try{const data=await json('/api/valley-sandbox/simulate?'+query);if(id===generation)draw(data);}
  catch(error){if(id===generation){el('status').textContent='Beregning feilet';el('error').textContent=error.message;}}
}
function schedule(){
  const id=++generation;clearTimeout(timer);
  el('temperature-value').textContent=number(el('temperature').value);
  timer=setTimeout(()=>simulate(id),150);
}
(async()=>{
  try{
    options=await json('/api/valley-sandbox/options');
    el('warning').textContent=options.warning;
    for(const s of options.stations)el('source').add(new Option(`${s.station_id} · ${s.name}`,s.station_id));
    for(const time of options.times)el('time').add(new Option(time.replace('T',' ').replace('+00:00',' UTC'),time));
    if(options.stations.some(s=>s.station_id==='SN55770'))el('source').value='SN55770';
    const testTime=options.times.find(t=>t.startsWith('2024-12-'));
    if(testTime)el('time').value=testTime;
    for(const id of ['source','time','hops','temperature'])el(id).disabled=false;
    map.fitBounds(options.stations.map(s=>[s.lat,s.lon]),{padding:[25,25]});
    el('temperature').addEventListener('input',schedule);
    for(const id of ['source','time','hops'])el(id).addEventListener('change',schedule);
    schedule();
  }catch(error){el('error').textContent=error.message;}
})();
