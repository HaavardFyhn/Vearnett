'use strict';
const el=id=>document.getElementById(id);
const map=L.map('map',{preferCanvas:true}).setView([61.4,6.4],8);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',{
  maxZoom:19,attribution:'© OpenStreetMap contributors'
}).addTo(map);
const cells=L.layerGroup().addTo(map),markers=L.layerGroup().addTo(map);
map.createPane('stationPane');map.getPane('stationPane').style.zIndex=650;
const stationRenderer=L.canvas({pane:'stationPane'});
const network=L.layerGroup().addTo(map);
const number=v=>Number(v).toLocaleString('nb-NO',{maximumFractionDigits:2});
let options=null,generation=0,timer=null,readyKey=null,loading=false,fitNext=true,lastGrid=null;
function color(value){
  const colors=[[5,48,97],[67,147,195],[247,247,247],[214,96,77],[103,0,31]];
  const p=Math.max(0,Math.min(4,(value+20)/10)),i=Math.min(3,Math.floor(p)),f=p-i;
  return `rgb(${colors[i].map((v,k)=>Math.round(v+(colors[i+1][k]-v)*f)).join(',')})`;
}
function radiusColor(value){
  const p=Math.max(0,Math.min(1,value/25));
  return `rgb(${[255,247,236].map((v,i)=>Math.round(v+([127,0,0][i]-v)*p)).join(',')})`;
}
function updateLegend(){
  const radius=el('display').value==='interval_radius_c';
  const div=legend.getContainer();
  div.replaceChildren();
  const title=document.createElement('div');
  title.textContent=radius?'Eksperimentell radius ± °C':'Eksperimentell temperatur °C';
  const ramp=document.createElement('div');ramp.className='ramp';
  if(radius)ramp.style.background='linear-gradient(to right,#fff7ec,#7f0000)';
  const scale=document.createElement('div');scale.textContent=radius?'0 — 25+ °C':'−20 — 0 — +20 °C';
  div.append(title,ramp,scale);
}
const legend=L.control({position:'bottomright'});
legend.onAdd=()=>{
  const div=L.DomUtil.create('div','legend');
  div.innerHTML='Eksperimentell temperatur &deg;C<div class="ramp"></div>-20 &nbsp;&nbsp; 0 &nbsp;&nbsp; +20<br>Farger mettes utenfor skalaen';
  return div;
};legend.addTo(map);
async function json(url,method='GET'){
  const response=await fetch(url,{method}),data=await response.json();
  if(!response.ok)throw new Error(data.error||`HTTP ${response.status}`);
  return data;
}
function query(){
  const q=new URLSearchParams({source:el('source').value,day:el('day').value,hour:el('hour').value});
  if(el('synthetic').checked)q.set('temperature',el('temperature').value);
  if(el('uncertainty').checked)q.set('uncertainty','1');
  return q;
}
function clear(){
  lastGrid=null;
  cells.clearLayers();markers.clearLayers();el('summary').textContent='';el('weather').textContent='';
  el('export').setAttribute('aria-disabled','true');el('export').removeAttribute('href');
}
function draw(data,q){
  clear();
  lastGrid={data,q};
  updateLegend();
  const radius=el('display').value==='interval_radius_c';
  const layer=L.geoJSON(data,{
    style:feature=>({stroke:false,fillColor:radius?radiusColor(feature.properties.interval_radius_c):color(feature.properties.temperature_c),fillOpacity:.8}),
    onEachFeature:(feature,polygon)=>{
      const div=document.createElement('span');
      const p=feature.properties;
      div.textContent=`${number(p.temperature_c)} °C · eksperimentell 1 km-celle`+
        (p.interval_radius_c!==undefined?` · ±${number(p.interval_radius_c)} °C · [${number(p.lower_c)}, ${number(p.upper_c)}] °C · ikke kalibrert kartdekning`:'');
      polygon.bindPopup(div);
    }
  }).addTo(cells);
  const meta=data.metadata,station=options.stations.find(s=>s.station_id===meta.source);
  const marker=L.circleMarker([station.lat,station.lon],{
    radius:8,color:'#111',weight:3,fillColor:color(meta.source_temperature_c),fillOpacity:1
  }).addTo(markers);
  const label=document.createElement('span');
  label.textContent=`${station.name}: ${number(meta.source_temperature_c)} °C · ${el('synthetic').checked?'syntetisk':'observert kilde'}`;
  marker.bindTooltip(label,{permanent:true});
  if(fitNext){map.fitBounds(layer.getBounds(),{padding:[20,20]});fitNext=false;}
  el('status').textContent=`${meta.cells} landceller · ${meta.time} · ${meta.model}`;
  el('summary').textContent=`Kilde ${number(meta.source_temperature_c)} °C · kart ${number(meta.minimum_c)} til ${number(meta.maximum_c)} °C`;
  el('observation').textContent=meta.observed_source_c===null?
    'Ingen QC-kildeobservasjon i arkivet for denne timen.':
    `Arkivert QC-kildeobservasjon: ${number(meta.observed_source_c)} °C`;
  el('weather').textContent=`MEPS utstedt ${meta.run_time}. Ingen kalibrert usikkerhet.`;
  el('source-warning').textContent=meta.source_extrapolation?
    `Kilde utenfor opprinnelig treningsnett (${meta.source_group}). Temperatur og intervaller er ytterligere ekstrapolasjon.`:'';
  if(meta.interval_radius_median_c!==undefined){
    el('summary').textContent+=` · median radius ±${number(meta.interval_radius_median_c)} °C`;
  }
  el('export').href='/api/temperature-map/export?'+q;
  el('export').setAttribute('aria-disabled','false');
}
async function render(id){
  if(readyKey!==`${el('source').value}|${el('day').value}`)return;
  clear();el('error').textContent='';el('status').textContent='Beregner temperaturer ...';
  const q=query();
  try{const data=await json('/api/temperature-map/grid?'+q);if(id===generation)draw(data,q);}
  catch(error){if(id===generation){el('status').textContent='Kartberegning feilet';el('error').textContent=error.message;}}
}
function schedule(){
  if(!el('uncertainty').checked)el('display').value='temperature_c';
  el('display').options[1].disabled=!el('uncertainty').checked;
  el('temperature').disabled=!el('synthetic').checked||loading;
  el('temperature-value').textContent=number(el('temperature').value);
  const id=++generation;clearTimeout(timer);
  clear();timer=setTimeout(()=>render(id),200);
}
function drawStations(){
  network.clearLayers();
  for(const station of options.stations){
    const marker=L.circleMarker([station.lat,station.lon],{
      renderer:stationRenderer,radius:4,color:station.group==='train'?'#154360':'#666',
      weight:1,fillColor:station.group==='train'?'#fff':'#bbb',fillOpacity:.9
    }).addTo(network);
    const label=document.createElement('span');
    label.textContent=`${station.station_id} · ${station.name} · ${station.group} · klikk for å velge`;
    marker.bindTooltip(label);
    marker.on('click',async()=>{
      if(loading){el('error').textContent='Vent til kartforberedelsen er ferdig før du velger en annen stasjon.';return;}
      el('source').value=station.station_id;changedArea();
      const id=generation;
      try{
        const info=await json('/api/temperature-map/station?'+query());
        if(id!==generation)return;
        el('source-warning').textContent=info.source_extrapolation?'Denne kilden var ikke i treningsnettet. Temperatur og intervaller er ekstrapolasjon.':'';
        el('observation').textContent=info.observed_source_c===null?
          'Ingen valgt QC-sensorobservasjon for denne stasjonen/timen. Velg annen tid eller aktiver syntetisk kilde.':
          `Arkivert QC-kildeobservasjon: ${number(info.observed_source_c)} °C`;
        map.panTo([station.lat,station.lon]);
        if(info.observed_source_c===null&&!el('synthetic').checked){
          el('status').textContent='Stasjon valgt; kildeobservasjon mangler.';return;
        }
        await prepare();
      }catch(error){if(id===generation)el('error').textContent=error.message;}
    });
  }
}
function changedArea(){
  ++generation;clearTimeout(timer);readyKey=null;fitNext=true;clear();
  el('observation').textContent='QC-kildeobservasjon er ikke kontrollert for det nye valget.';
  el('source-warning').textContent='';
  el('status').textContent='Trykk Last og vis for valgt stasjon/dato.';
  el('error').textContent='';
}
async function prepare(){
  if(loading)return;
  const id=++generation;clearTimeout(timer);clear();loading=true;el('error').textContent='';
  for(const name of ['source','day','hour','synthetic','temperature','prepare','uncertainty','display'])el(name).disabled=true;
  const key=`${el('source').value}|${el('day').value}`;
  try{
    const q=new URLSearchParams({source:el('source').value,day:el('day').value});
    let state=await json('/api/temperature-map/prepare?'+q,'POST');
    while(state.state==='running'){
      el('status').textContent=state.detail||'Laster ...';
      await new Promise(resolve=>setTimeout(resolve,1000));
      state=await json('/api/temperature-map/status');
      if(`${state.source}|${state.day}`!==key)throw new Error('Kartforberedelsen ble endret av en annen forespørsel.');
    }
    if(state.state!=='ready')throw new Error(state.error||'Kartforberedelse feilet');
    readyKey=key;await render(id);
  }catch(error){el('status').textContent='Kartforberedelse feilet';el('error').textContent=error.message;}
  finally{
    loading=false;
    for(const name of ['source','day','hour','synthetic','prepare'])el(name).disabled=false;
    el('uncertainty').disabled=!options.uncertainty.available;el('display').disabled=false;
    el('temperature').disabled=!el('synthetic').checked;
  }
}
(async()=>{
  try{
    options=await json('/api/temperature-map/options');el('warning').textContent=options.warning;
    for(const s of options.stations)el('source').add(new Option(`${s.station_id} · ${s.name} · ${s.group}`,s.station_id));
    drawStations();
    const available=options.uncertainty.available;
    el('uncertainty').disabled=!available;el('uncertainty').checked=available;
    el('display').options[1].disabled=!available;
    el('uncertainty-status').textContent=available?
      `Eksperimentelle stasjonsbaserte intervaller. Test: ${number(options.uncertainty.test.night_coverage*100)} % nattdekning (mål 90 %), median radius ±${number(options.uncertainty.test.radius_median_c)} °C. Ingen 90 %-garanti for kartceller, syntetiske kilder eller nye kildestasjoner.`:
      'Verifisert usikkerhetsmodell mangler. Intervaller kan ikke beregnes.';
    el('uncertainty').addEventListener('change',schedule);
    el('display').addEventListener('change',()=>{if(lastGrid)draw(lastGrid.data,lastGrid.q);else updateLegend();});
    el('show-stations').addEventListener('change',()=>{if(el('show-stations').checked)network.addTo(map);else map.removeLayer(network);});
    el('national').disabled=false;
    el('national').addEventListener('click',()=>map.fitBounds(options.stations.map(s=>[s.lat,s.lon]),{padding:[20,20]}));
    for(const day of options.days)el('day').add(new Option(day,day));
    if(options.stations.some(s=>s.station_id==='SN55770'))el('source').value='SN55770';
    if(options.days.includes('2024-12-15'))el('day').value='2024-12-15';
    for(const name of ['source','day','hour','synthetic','prepare'])el(name).disabled=false;
    for(const name of ['source','day'])el(name).addEventListener('change',changedArea);
    for(const name of ['hour','synthetic'])el(name).addEventListener('change',schedule);
    el('temperature').addEventListener('input',schedule);
    el('prepare').addEventListener('click',prepare);
    await prepare();
  }catch(error){el('error').textContent=error.message;}
})();
