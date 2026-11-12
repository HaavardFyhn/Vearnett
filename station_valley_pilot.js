'use strict';
const el=id=>document.getElementById(id);
const map=L.map('map').setView([61.4,6.4],9);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',{
  maxZoom:19,attribution:'© OpenStreetMap contributors'
}).addTo(map);
const layer=L.layerGroup().addTo(map);
let options=null,current=null,timeline=null,requestId=0;
const methodsColors={forecast_transfer:'#866019',terrain:'#3383b5',regime:'#c0333f',graph_regime:'#734998'};
const num=value=>Number(value).toLocaleString('nb-NO',{maximumFractionDigits:2});
const station=id=>options.stations.find(item=>item.station_id===id);
async function json(url){
  const response=await fetch(url),payload=await response.json();
  if(!response.ok)throw new Error(payload.error||`HTTP ${response.status}`);
  return payload;
}
function query(extra={}){
  return new URLSearchParams({target:el('target').value,condition:el('condition').value,...extra});
}
function busy(value){
  for(const id of ['target','method','condition','time','source','time-slider'])el(id).disabled=value||!options||!options.times.length;
}
function color(error){
  const colors=[[5,48,97],[67,147,195],[247,247,247],[214,96,77],[103,0,31]];
  const position=Math.max(0,Math.min(4,(error+6)/3)),index=Math.min(3,Math.floor(position)),fraction=position-index;
  return `rgb(${colors[index].map((value,k)=>Math.round(value+(colors[index+1][k]-value)*fraction)).join(',')})`;
}
function tableSummary(){
  const table=document.createElement('table'),head=document.createElement('tr');
  for(const label of ['Metode','RMSE °C','MAE °C']){const th=document.createElement('th');th.textContent=label;head.append(th);}
  table.append(head);
  for(const method of options.methods){
    const score=options.summary[method.id];if(!score)continue;
    const row=document.createElement('tr');row.classList.toggle('active',method.id===el('method').value);
    for(const value of [method.label,num(score.rmse_c),num(score.mae_c)]){
      const cell=document.createElement('td');cell.textContent=value;row.append(cell);
    }table.append(row);
  }
  el('summary').replaceChildren(table);
  const selected=options.methods.find(item=>item.id===options.selected_by_validation);
  el('summary-info').textContent=`Filter: ${options.rows} kilde–mål-rader på ${options.nights} netter. Radene er ikke uavhengige. Valgt på validering: ${selected.label}; metodevelgeren er kun for sammenligning, ikke ny modellseleksjon.`;
}
function drawCase(){
  layer.clearLayers();el('transfers').replaceChildren();
  tableSummary();
  if(!current)return;
  const target=station(current.target);
  L.circleMarker([target.lat,target.lon],{radius:10,color:'#111',fillColor:'#fff',fillOpacity:1,weight:3})
    .bindTooltip(`${target.station_id} · ${target.name} · observert ${num(current.observed_c)} °C`).addTo(layer);
  const method=el('method').value;
  el('case-info').textContent=`${current.time.replace('T',' ').replace('+00:00',' UTC')}\n${target.name}: observert ${num(current.observed_c)} °C\n${current.transfers.length} separate kildeoverføringer`;
  for(const row of current.transfers){
    const source=station(row.source),prediction=row[method+'_prediction_c'],error=prediction-current.observed_c;
    const line=L.polyline([[source.lat,source.lon],[target.lat,target.lon]],{color:color(error),weight:3,opacity:.85})
      .bindTooltip(`${source.station_id} → ${target.station_id}: pred. ${num(prediction)} °C · feil ${num(error)} °C`).addTo(layer);
    const marker=L.circleMarker([source.lat,source.lon],{radius:7,color:'#333',weight:1,fillColor:color(error),fillOpacity:1})
      .bindTooltip(`${source.station_id} · ${source.name} · kilde observert ${num(row.source_c)} °C`).addTo(layer);
    const select=()=>{el('source').value=row.source;loadTimeline();};
    line.on('click',select);marker.on('click',select);
    const tr=document.createElement('tr'),td=document.createElement('td'),button=document.createElement('button');
    button.textContent=row.source;button.title=source.name;button.onclick=select;td.append(button);tr.append(td);
    for(const value of [num(prediction),num(error)]){const cell=document.createElement('td');cell.textContent=value;tr.append(cell);}
    el('transfers').append(tr);
  }
  el('interval').textContent=`Kalibrert 90 %-radius for valgt metode: ±${num(options.intervals[method])} °C. Kalibrert på maksimale feil per valideringsnatt; ingen garanti ved geografisk fordelingsskift. Ikke en lokal usikkerhetsflate.`;
}
function svg(tag,attrs,text){
  const node=document.createElementNS('http://www.w3.org/2000/svg',tag);
  for(const [key,value] of Object.entries(attrs))node.setAttribute(key,value);
  if(text!==undefined)node.textContent=text;return node;
}
function drawChart(){
  el('chart').replaceChildren();el('chart-key').replaceChildren();
  if(!timeline||!timeline.points.length){el('source-score').textContent='Ingen sammenlignbare tidsseriepunkter for denne kilden/filteret.';return;}
  const points=timeline.points,series=[{id:'observed',label:'Observert',color:'#111'},
    ...options.methods.map(method=>({...method,color:methodsColors[method.id]}))];
  const values=points.flatMap(p=>[p.target_c,...options.methods.map(m=>p[m.id+'_prediction_c'])]);
  const lo=Math.floor(Math.min(...values)-1),hi=Math.ceil(Math.max(...values)+1);
  const first=Date.parse(points[0].time),last=Date.parse(points[points.length-1].time);
  const x=p=>55+755*(Date.parse(p.time)-first)/Math.max(last-first,1),y=v=>200-175*(v-lo)/(hi-lo);
  for(let i=0;i<=4;i++){
    const value=lo+(hi-lo)*i/4;
    el('chart').append(svg('line',{x1:55,x2:810,y1:y(value),y2:y(value),stroke:'#ddd'}),
      svg('text',{x:48,y:y(value)+4,'text-anchor':'end','font-size':11},num(value)+'°'));
  }
  const chartTitle=svg('title',{},`${timeline.source} til ${timeline.target}: ${points.length} sammenlignbare tidspunkter`);
  el('chart').append(chartTitle);
  for(const entry of series){
    const key=document.createElement('span');key.style.color=entry.color;key.textContent='● '+entry.label;el('chart-key').append(key);
    let previous=null;
    for(const point of points){
      const value=entry.id==='observed'?point.target_c:point[entry.id+'_prediction_c'];
      if(previous&&Date.parse(point.time)-Date.parse(previous.point.time)<=18*3600000){
        el('chart').append(svg('line',{x1:x(previous.point),x2:x(point),y1:y(previous.value),y2:y(value),stroke:entry.color,'stroke-width':entry.id==='observed'?2:1,opacity:.7}));
      }
      const dot=svg('circle',{cx:x(point),cy:y(value),r:entry.id==='observed'?3:2,fill:entry.color,tabindex:0});
      dot.append(svg('title',{},`${point.time} · ${entry.label}: ${num(value)} °C`));
      const choose=()=>{if(options.times.includes(point.time)){el('time').value=point.time;loadCase();}};
      dot.onclick=choose;dot.onkeydown=event=>{if(event.key==='Enter')choose();};el('chart').append(dot);
      previous={point,value};
    }
  }
  for(const [stamp,anchor,offset] of [[first,'start',55],[last,'end',810]]){
    el('chart').append(svg('text',{x:offset,y:224,'text-anchor':anchor,'font-size':11},new Date(stamp).toISOString().slice(0,10)));
  }
  const selectedX=55+755*(Date.parse(el('time').value)-first)/Math.max(last-first,1);
  if(selectedX>=55&&selectedX<=810)el('chart').append(svg('line',{x1:selectedX,x2:selectedX,y1:20,y2:204,stroke:'#666','stroke-dasharray':'3,3'}));
  const score=timeline.summary[el('method').value];
  el('source-score').textContent=`Valgt kilde ${timeline.source} · ${score.nights} netter\nRMSE ${num(score.rmse_c)} °C · bias ${num(score.bias_c)} °C (valgt metode/filter).`;
}
async function loadTimeline(){
  const source=el('source').value,target=el('target').value,condition=el('condition').value;
  try{
    const result=await json(`/api/valley-pilot/timeline?${query({source})}`);
    if(source!==el('source').value||target!==el('target').value||condition!==el('condition').value)return;
    timeline=result;drawChart();
  }catch(error){el('error').textContent=error.message;}
}
async function loadCase(){
  const token=++requestId;busy(true);el('error').textContent='';
  try{
    const result=await json(`/api/valley-pilot/case?${query({time:el('time').value})}`);
    if(token!==requestId)return;
    current=result;
    el('time-slider').value=options.times.indexOf(current.time);
    drawCase();drawChart();el('status').textContent='Klar · faktiske holdt-utenfor testresultater';
  }catch(error){el('error').textContent=error.message;el('status').textContent='Tidspunktet kunne ikke lastes';}
  finally{if(token===requestId)busy(false);}
}
async function loadOptions(initial=false){
  ++requestId;busy(true);el('error').textContent='';
  try{
    const result=await json(initial?'/api/valley-pilot/options':`/api/valley-pilot/options?${query()}`);
    options=result;current=null;timeline=null;
    if(initial){
      for(const id of result.targets)el('target').add(new Option(`${id} · ${station(id).name}`,id));
      for(const method of result.methods)el('method').add(new Option(method.label,method.id));
      el('method').value=result.selected_by_validation;
      el('limitations').replaceChildren(...result.limitations.map(text=>{const li=document.createElement('li');li.textContent=text;return li;}));
      const coordinates=result.stations.map(s=>[s.lat,s.lon]);map.fitBounds(coordinates,{padding:[30,30]});
    }
    const oldSource=el('source').value;
    el('source').replaceChildren(...result.sources.map(id=>new Option(`${id} · ${station(id).name}`,id)));
    if(result.sources.includes(oldSource))el('source').value=oldSource;
    el('time').replaceChildren(...result.times.map(time=>new Option(time.replace('T',' ').replace('+00:00',' UTC'),time)));
    el('time-slider').max=Math.max(0,result.times.length-1);el('time-slider').value=0;
    if(!result.times.length){layer.clearLayers();el('transfers').replaceChildren();el('case-info').textContent='Ingen støttede tidspunkter i dette filteret.';tableSummary();drawChart();el('status').textContent='Ingen data';busy(false);el('condition').disabled=false;el('target').disabled=false;return;}
    await loadTimeline();await loadCase();
  }catch(error){el('error').textContent=error.message;el('status').textContent='Pilotresultatene kunne ikke lastes';busy(false);}
}
el('method').onchange=()=>{drawCase();drawChart();};
el('target').onchange=()=>loadOptions();
el('condition').onchange=()=>loadOptions();
el('source').onchange=()=>loadTimeline();
el('time').onchange=()=>loadCase();
el('time-slider').onchange=()=>{el('time').value=options.times[Number(el('time-slider').value)];loadCase();};
loadOptions(true);
