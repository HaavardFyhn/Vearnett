'use strict';
const el=id=>document.getElementById(id);
const map=L.map('map',{preferCanvas:true}).setView([63,10],5);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',{
  maxZoom:19,attribution:'© OpenStreetMap contributors'
}).addTo(map);
const targets=L.layerGroup().addTo(map),sources=L.layerGroup().addTo(map);
const number=v=>Number(v).toLocaleString('en-GB',{maximumFractionDigits:2});
let options=null,current=null,version=0,fitNext=true,selectionVersion=0;
const labels={
  prediction_c:'Predicted temperature (mean of evaluations if multiple sources)',
  target_c:'Observed target temperature',
  bias_c:'Signed error: prediction minus observation',
  rmse_c:'Actual RMSE across evaluated source rows',
  predicted_pooled_rmse_c:'Predicted RMSE (NOT interval radius)',
  radius_c:'Joint interval radius ± °C (NOT RMSE)',
  individual_coverage:'Observed interval coverage'
};
function query(){
  const q=new URLSearchParams({model:el('model').value,split:el('split').value,date:el('date').value});
  if(el('model').value==='joint')q.set('hour',el('hour').value);
  else{q.set('variable',el('variable').value);if(el('source').value)q.set('source',el('source').value);}
  return q;
}
async function json(url){
  const response=await fetch(url),data=await response.json();
  if(!response.ok)throw new Error(data.error||`HTTP ${response.status}`);
  return data;
}
function text(tag,value){const node=document.createElement(tag);node.textContent=value;return node;}
function table(container,headers,rows){
  const t=document.createElement('table'),head=document.createElement('thead'),body=document.createElement('tbody');
  const h=document.createElement('tr');for(const value of headers)h.append(text('th',value));head.append(h);
  for(const values of rows){const row=document.createElement('tr');for(const value of values)row.append(text('td',value));body.append(row);}
  t.append(head,body);container.append(t);return body;
}
function showMetrics(container,value){
  container.replaceChildren();
  const rows=[['Rows / targets',`${value.rows.toLocaleString()} / ${value.targets.toLocaleString()}`],
    ['Actual temperature RMSE',`${number(value.rmse_c)} °C`],['MAE',`${number(value.mae_c)} °C`],
    ['Bias (predicted − observed)',`${number(value.bias_c)} °C`]];
  if(value.predicted_pooled_rmse_c!==undefined)rows.push(['Predicted pooled RMSE',`${number(value.predicted_pooled_rmse_c)} °C`]);
  if(value.interval_radius_median_c!==undefined)rows.push(['Median interval radius',`±${number(value.interval_radius_median_c)} °C`],
    ['Individual interval coverage',`${number(value.individual_coverage*100)}%`]);
  table(container,['Measure','Value'],rows);
}
function modelChanged(){
  ++version;selectionVersion++;targets.clearLayers();sources.clearLayers();current=null;fitNext=true;
  const key=el('model').value,data=options.models[key],joint=key==='joint';
  el('hour-control').hidden=!joint;el('variable-control').hidden=joint;
  el('warning').textContent=data.warning;
  el('split').replaceChildren(...data.splits.map(value=>new Option(value==='calibration'?'Assessment 2019–20 (not tuning)':value,value)));
  el('split').value='test';
  const metrics=['prediction_c','target_c','bias_c','rmse_c',...(joint?['radius_c','individual_coverage']:['predicted_pooled_rmse_c'])];
  el('metric').replaceChildren(...metrics.map(value=>new Option(labels[value],value)));
  el('metric').value=joint?'radius_c':'predicted_pooled_rmse_c';
  el('source').replaceChildren(new Option('All evaluated sources (summary, NOT fusion)',''),
    ...data.sources.map(value=>new Option(`Cell ${value}`,value)));
  showMetrics(el('overall'),data.overall);
  const selected=joint?'Frozen validation-selected six-source fusion':data.report.selected.mean;
  el('overall').append(text('p',`Temperature model: ${selected}.`));
  if(joint){
    const r=data.report;
    el('overall').append(text('p',`Full-group night coverage: ${number(r.full_joint_test.night_coverage*100)}% (${r.full_joint_test.nights} nights). Full + deletion night coverage: ${number(r.full_and_deletions_test.night_coverage*100)}%. Target: 90%.`));
  }
  diagnostics(key,data.report);
  datesChanged();
}
function diagnostics(key,report){
  const container=el('diagnostics');container.replaceChildren();
  if(key==='joint'){
    table(container,['Test region','Night coverage','Median radius ± °C'],
      Object.entries(report.full_joint_test.block).map(([name,v])=>[name,`${number(v.night_coverage*100)}%`,number(v.radius_median_c)]));
  }else{
    const r=report.assessments.test;
    table(container,['Temperature baseline','Actual test RMSE °C'],
      [['Copy source',number(r.raw_difference.rmse_c)],['Learned offset',number(r.offset.rmse_c)],['Ridge',number(r.ridge.rmse_c)],['Selected neural model',number(r.temperature.rmse_c)]]);
    table(container,['Test region','Actual RMSE °C','Predicted RMSE °C'],
      Object.entries(r.error_model.block).map(([name,v])=>[name,number(v.observed_rmse_c),number(v.predicted_pooled_rmse_c)]));
    table(container,['Predicted-error quintile','Actual RMSE °C','Predicted RMSE °C'],
      r.error_model.predicted_rmse_bins.map((v,i)=>[i+1,number(v.observed_rmse_c),number(v.predicted_pooled_rmse_c)]));
  }
}
function datesChanged(){
  const key=el('model').value,dates=options.models[key].dates[el('split').value],old=el('date').value;
  el('date').replaceChildren(...dates.map(value=>new Option(value,value)));
  el('date').value=dates.includes(old)?old:(dates.includes(options.default_date[key])?options.default_date[key]:dates[0]);
  fitNext=true;load();
}
function colour(value,metric){
  if(metric==='individual_coverage')return value>=1?'#238b45':'#cb181d';
  const signed=['prediction_c','target_c','bias_c'].includes(metric);
  const limit=metric==='bias_c'?5:20;
  if(signed){
    const p=Math.max(-1,Math.min(1,value/limit));
    const end=p<0?[5,48,97]:[178,24,43],amount=Math.abs(p);
    return `rgb(${[247,247,247].map((v,i)=>Math.round(v+(end[i]-v)*amount)).join(',')})`;
  }
  const p=Math.max(0,Math.min(1,value/(metric==='radius_c'?20:5)));
  return `rgb(${[255,247,236].map((v,i)=>Math.round(v+([127,0,0][i]-v)*p)).join(',')})`;
}
const legend=L.control({position:'bottomright'});
legend.onAdd=()=>L.DomUtil.create('div','legend');legend.addTo(map);
function updateLegend(){
  const metric=el('metric').value,container=legend.getContainer(),signed=['prediction_c','target_c','bias_c'].includes(metric);
  container.replaceChildren(text('strong',labels[metric]));
  const ramp=text('div','');ramp.className='ramp';
  ramp.style.background=metric==='individual_coverage'?'linear-gradient(to right,#cb181d,#238b45)':
    signed?'linear-gradient(to right,#053061,#f7f7f7,#b2182b)':'linear-gradient(to right,#fff7ec,#7f0000)';
  const scale=metric==='individual_coverage'?'Outside / inside observed interval':
    signed?(metric==='bias_c'?'−5 — 0 — +5 °C':'−20 — 0 — +20 °C'):(metric==='radius_c'?'0 — 20+ °C':'0 — 5+ °C');
  container.append(ramp,text('div',scale),text('small','Evaluated targets only; colours saturate.'));
}
function draw(){
  targets.clearLayers();updateLegend();
  if(!current)return;
  const metric=el('metric').value;
  const layer=L.geoJSON(current,{
    style:feature=>({color:'#555',weight:.5,fillOpacity:.85,fillColor:colour(feature.properties[metric],metric)}),
    pointToLayer:(feature,latlng)=>L.circleMarker(latlng,{radius:8,color:'#222',weight:1,
      fillOpacity:1,fillColor:colour(feature.properties[metric],metric)}),
    onEachFeature:(feature,shape)=>{
      const p=feature.properties;
      shape.bindTooltip(text('span',`${p.name} · ${number(p[metric])}${metric==='individual_coverage'?'':' °C'}`));
      shape.on('click',()=>detail(p));
    }
  }).addTo(targets);
  if(fitNext&&current.features.length){map.fitBounds(layer.getBounds(),{padding:[20,20],maxZoom:11});fitNext=false;}
}
async function load(){
  const id=++version;selectionVersion++;current=null;targets.clearLayers();sources.clearLayers();
  el('detail').textContent='Click an evaluated target.';el('summary').replaceChildren();el('aggregation').textContent='';
  el('target').replaceChildren(new Option('Select a target to zoom and inspect',''));el('target').disabled=true;
  el('export').setAttribute('aria-disabled','true');el('export').removeAttribute('href');
  el('error').textContent='';el('status').textContent='Loading verified evaluation samples…';
  try{
    const q=query(),data=await json('/api/model-evaluation/grid?'+q);
    if(id!==version)return;
    current=data;showMetrics(el('summary'),data.summary);el('aggregation').textContent=data.aggregation;
    el('target').replaceChildren(new Option('Select a target to zoom and inspect',''),
      ...data.features.map(f=>new Option(`${f.properties.target} · ${f.properties.name}`,f.properties.target)));
    el('target').disabled=false;
    el('status').textContent=`${data.features.length} evaluated targets · ${data.summary.rows} rows · ${el('date').value}`;
    el('export').href='/api/model-evaluation/export?'+q;el('export').setAttribute('aria-disabled','false');draw();
  }catch(error){if(id===version){el('status').textContent='Evaluation unavailable for these filters';el('error').textContent=error.message;}}
}
async function detail(p){
  const id=++selectionVersion,generation=version;
  sources.clearLayers();el('detail').textContent='Loading source diagnostics…';
  el('target').value=p.target;
  try{
    const q=query();q.set('target',p.target);
    const data=await json('/api/model-evaluation/detail?'+q);
    if(id!==selectionVersion||generation!==version)return;
    const container=el('detail');container.replaceChildren(text('strong',p.name));
    showMetrics(container,p);container.prepend(text('strong',`${p.name} · target ${p.target}`));
    container.append(text('p',`Observed ${number(p.target_c)} °C; predicted ${number(p.prediction_c)} °C.`));
    if(p.radius_c!==undefined)container.append(text('p',`Full interval ±${number(p.radius_c)} °C: [${number(p.lower_c)}, ${number(p.upper_c)}] °C · ${p.source_count} sources.`));
    container.append(text('p',data.explanation));
    if(el('model').value==='joint')container.append(text('p','Source colours: green = radius increases without the reading; purple = radius decreases without it. Changes are without minus full, not station quality.'));
    if(data.sole_source_no_deletion){
      container.append(text('p','Only one source. Removing it produces NoData, not a zero-radius prediction. No deletion prediction was evaluated.'));
      for(const s of data.sources){
        container.append(text('p',`Sole source: ${s.source} · ${s.name} · observed ${number(s.source_c)} °C.`));
        L.circleMarker([s.lat,s.lon],{radius:7,color:'#111',fillOpacity:1,fillColor:'#2171b5'})
          .bindTooltip(text('span',`${s.source} · ${s.name} · sole source`)).addTo(sources);
      }
      return;
    }
    const joint=el('model').value==='joint';
    const header=joint?['Removed source','Reading °C','Radius without ± °C','Δ radius °C','Δ temperature °C']:
      ['Source cell','Source °C','Prediction °C','Error °C','Predicted RMSE °C'];
    const rows=data.sources.map(s=>joint?[`${s.source} ${s.name}`,number(s.source_c),number(s.radius_c),number(s.radius_change_c),number(s.temperature_change_c)]:
      [s.source,number(s.source_c),number(s.prediction_c),number(s.error_c),number(s.predicted_rmse_c)]);
    const body=table(container,header,rows);
    const target=current.features.find(f=>f.properties.target===p.target);
    const center=target.geometry.type==='Point'?target.geometry.coordinates:
      [target.geometry.coordinates[0].slice(0,4).reduce((sum,v)=>sum+v[0],0)/4,
       target.geometry.coordinates[0].slice(0,4).reduce((sum,v)=>sum+v[1],0)/4];
    data.sources.forEach((s,i)=>{
      const marker=L.circleMarker([s.lat,s.lon],{radius:7,color:'#111',fillOpacity:1,
        fillColor:joint?(s.radius_change_c>=0?'#238b45':'#756bb1'):'#2171b5'}).addTo(sources);
      marker.bindTooltip(text('span',`${s.source} · ${s.name}`));
      L.polyline([[center[1],center[0]],[s.lat,s.lon]],{weight:1,color:'#555',dashArray:'4 4'}).addTo(sources);
      const row=body.children[i];row.className='source';row.tabIndex=0;
      const focus=()=>{map.panTo([s.lat,s.lon]);marker.openTooltip();};
      row.addEventListener('click',focus);row.addEventListener('keydown',event=>{if(event.key==='Enter')focus();});
    });
  }catch(error){if(id===selectionVersion&&generation===version){el('detail').textContent='Source diagnostics failed';el('error').textContent=error.message;}}
}
el('model').addEventListener('change',modelChanged);
el('split').addEventListener('change',datesChanged);
for(const id of ['date','hour','variable','source'])el(id).addEventListener('change',load);
el('metric').addEventListener('change',draw);
el('target').addEventListener('change',()=>{
  const feature=current?.features.find(f=>f.properties.target===el('target').value);
  if(!feature)return;
  const p=feature.properties,center=feature.geometry.type==='Point'?feature.geometry.coordinates:
    [feature.geometry.coordinates[0].slice(0,4).reduce((sum,v)=>sum+v[0],0)/4,
     feature.geometry.coordinates[0].slice(0,4).reduce((sum,v)=>sum+v[1],0)/4];
  map.setView([center[1],center[0]],el('model').value==='joint'?10:13);
  detail(p);
});
el('national').addEventListener('click',()=>{fitNext=true;draw();});
json('/api/model-evaluation/options').then(data=>{
  options=data;for(const id of ['model','split','date'])el(id).disabled=false;modelChanged();
}).catch(error=>{el('status').textContent='Verified results could not be loaded';el('error').textContent=error.message;});
