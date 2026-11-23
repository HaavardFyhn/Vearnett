// Run with: node test_station_saving.js
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const source=fs.readFileSync('station_app.js','utf8');
const template=fs.readFileSync('station_app.html','utf8');
const cutoffSlider={value:'0',max:'0',setAttribute(){}};
const cutoffSwatch={style:{},setAttribute(){}};
const cutoffContext=vm.createContext({
  state:{dataset:'temperature:senorge30',layers:['temperature:senorge30'],metric:'RMSE',combine:'sum',cell_m:500,revision:1},
  mode:'placement',placementCatalogs:new Map([[JSON.stringify(['temperature:senorge30','svv']),{maximum:99}]]),
  selectedPlacementNetwork:()=> 'svv',getWeatherRoadSettings:()=>null,
  element:id=>id==='rmse-min'?cutoffSlider:cutoffSwatch,
  roadErrorColour:()=> '#000',uncertaintyScaleColour:()=> '#000',
  updateMap:()=>{},error:message=>{throw Error(message);},
  fetch:async()=>({ok:true,json:async()=>({minimum:0,maximum:6})})
});
vm.runInContext(source.slice(source.indexOf('let combinedCutoffRange='),
  source.indexOf('let rmseTimer=')),cutoffContext);
vm.runInContext(source.slice(source.indexOf('function placementScaleColour('),
  source.indexOf('const ROAD_SCORE_COLOURS=')),cutoffContext);
(async()=>{
  await cutoffContext.refreshRmseRange();
  assert.equal(cutoffSlider.disabled,false);
  assert.equal(cutoffSlider.max,'1');
  cutoffContext.renderUncertaintyCutoff();
  assert.equal(cutoffSwatch.style.background,'white');
  cutoffSlider.value='0.5';
  assert.ok(Math.abs(cutoffContext.rmseMinimum()-9)<1e-10);
  cutoffContext.renderUncertaintyCutoff();
  assert.equal(cutoffSwatch.style.background,'rgb(252,140,59)');
  cutoffContext.mode='uncertainty';
  await cutoffContext.refreshRmseRange();
  assert.equal(cutoffSlider.value,'0');
  cutoffSlider.value='3';
  assert.equal(cutoffContext.rmseMinimum(),3);
  cutoffContext.mode='reduction';
  await cutoffContext.refreshRmseRange();
  cutoffContext.renderUncertaintyCutoff();
  assert.equal(cutoffSlider.disabled,true);
  assert.equal(cutoffContext.rmseMinimum(),0);
  assert.equal(cutoffSwatch.title,'Ikke tilgjengelig for Endring i usikkerhet');
  cutoffContext.mode='uncertainty';
  await cutoffContext.refreshRmseRange();
  assert.equal(cutoffSlider.disabled,false);
  assert.equal(cutoffContext.rmseMinimum(),3);
  cutoffSlider.value='0';
  cutoffContext.renderUncertaintyCutoff();
  assert.equal(cutoffSwatch.style.background,'black');
  cutoffContext.mode='placement';
  await cutoffContext.refreshRmseRange();
  assert.equal(cutoffSlider.value,'0');
  cutoffContext.state.layers.push('snow:class');
  await cutoffContext.refreshRmseRange();
  cutoffSlider.value='0.5';
  assert.equal(cutoffContext.rmseMinimum(),3);
})().catch(error=>{console.error(error);process.exitCode=1;});
assert.ok(source.includes('${placementQuery}${corridorQuery}${rmseQuery}`'));
const imageRequests=[],removedImages=[],imageErrors=[],finishedImages=[];
const queueContext=vm.createContext({
  overlay:null,overlaySerial:0,overlayKey:'current',
  beginLoading:label=>label,finishLoading:(token,failed)=>finishedImages.push([token,failed]),
  overlayOpacity:()=>0.7,error:message=>imageErrors.push(message),
  map:{removeLayer:layer=>removedImages.push(layer.url)},
  L:{imageOverlay:(url,bounds)=>{const layer={url,bounds,handlers:{},addTo(){imageRequests.push(this);return this;},
    once(event,handler){this.handlers[event]=handler;},setOpacity(value){this.opacity=value;}};
    return layer;}}
});
vm.runInContext(source.slice(source.indexOf('let activeMapImage='),
  source.indexOf('function updateMap()')),queueContext);
queueContext.overlaySerial=1;
queueContext.queueMapImage('first',[],1,'first','first error');
queueContext.overlaySerial=2;
queueContext.queueMapImage('middle',[],2,'middle','middle error');
queueContext.overlaySerial=3;
queueContext.queueMapImage('latest',[],3,'latest','latest error');
assert.deepEqual(imageRequests.map(item=>item.url),['first']);
imageRequests[0].handlers.load();
assert.deepEqual(imageRequests.map(item=>item.url),['first','latest']);
assert.deepEqual(removedImages,['first']);
imageRequests[1].handlers.load();
assert.equal(queueContext.overlay.url,'latest');
assert.equal(queueContext.overlay.opacity,0.7);
assert.deepEqual(imageErrors,[]);
queueContext.overlaySerial=4;
queueContext.queueMapImage('old',[],4,'old','old error');
queueContext.overlaySerial=5;
queueContext.queueMapImage('cancelled',[],5,'cancelled','cancelled error');
queueContext.cancelQueuedMapImage();
imageRequests[2].handlers.error();
assert.equal(imageRequests.length,3);
assert.deepEqual(imageErrors,[]);
queueContext.overlaySerial=6;
queueContext.queueMapImage('failed',[],6,'failed','visible error');
imageRequests[3].handlers.error();
assert.deepEqual(imageErrors,['visible error']);
assert.equal(queueContext.overlayKey,null);
assert.equal(finishedImages.at(-1)[1],true);
const tileRequests=[],tileFinished=[],tileErrors=[];
let paintFrame=null;
let tileNetwork='svv',tileCutoff=0;
const tileContext=vm.createContext({
  state:{dataset:'temperature:senorge30',revision:7,bounds:[[58,5],[71,31]],counties:null},
  overlay:null,overlaySerial:0,overlayKey:null,
  URLSearchParams,selectedPlacementNetwork:()=>tileNetwork,rmseMinimum:()=>tileCutoff,
  overlayOpacity:()=>0.7,beginLoading:()=>1,finishLoading:(...args)=>tileFinished.push(args),
  requestAnimationFrame:callback=>{paintFrame=callback;return 1;},cancelAnimationFrame:()=>{paintFrame=null;},
  map:{removeLayer:()=>{}},error:message=>tileErrors.push(message),
  L:{tileLayer:(url,options)=>{const layer={url,options,handlers:{},on(name,callback){this.handlers[name]=callback;return this;},
    addTo(){this.handlers.loading();return this;},setOpacity(){}};
    tileRequests.push(layer);return layer;}}
});
vm.runInContext(source.slice(source.indexOf('let activeMapImage='),
  source.indexOf('function updateMap()')),tileContext);
tileContext.createRasterTileLayer=tileContext.L.tileLayer;
const pixelSource=new Uint8ClampedArray([255,255,204,210,255,237,160,210,128,0,38,210,0,0,0,0]);
const pixelCanvas={rasterValues:new Float32Array([0,.5,1,NaN]),rasterColours:pixelSource,
  rasterPixels:{data:new Uint8ClampedArray(pixelSource)},getContext:()=>({putImageData(){}})};
tileContext.paintRasterCanvas(pixelCanvas,.5);
assert.deepEqual(Array.from(pixelCanvas.rasterPixels.data).filter((_,index)=>index%4===3),[0,0,210,0]);
tileContext.paintRasterCanvas(pixelCanvas,1);
assert.deepEqual(Array.from(pixelCanvas.rasterPixels.data).filter((_,index)=>index%4===3),[0,0,0,0]);
tileContext.paintRasterCanvas(pixelCanvas,0);
assert.deepEqual(Array.from(pixelCanvas.rasterPixels.data).filter((_,index)=>index%4===3),[0,210,210,0]);
tileContext.paintRasterCanvas(pixelCanvas,null);
assert.deepEqual(Array.from(pixelCanvas.rasterPixels.data),Array.from(pixelSource),'Switching the filter off restores exact original colours and alpha');
const tileMetadata={tile_version:'first',bounds:[[58,5],[71,31]]};
tileContext.showPlacementTiles(tileMetadata,{corridor:true,distance_m:500,bounds:[5,58,15,65]});
tileContext.showPlacementTiles(tileMetadata,{corridor:true,distance_m:500,bounds:[6,59,16,66]});
assert.equal(tileRequests.length,1,'Panning must retain the same tile layer and URLs');
tileRequests[0].handlers.load();
tileCutoff=2;
tileContext.showPlacementTiles(tileMetadata,{corridor:true,distance_m:500,bounds:[6,59,16,66]});
assert.equal(tileRequests.length,1,'Changing the cutoff must not request new tiles');
assert.equal(tileRequests[0].url.includes('rmse_min='),false);
assert.equal(tileRequests[0].options.opacity,0.7);
assert.equal(tileRequests[0].options.pane,'overlayPane','Placement colours must not inherit background grayscale');
tileNetwork='svv-met';
tileContext.showPlacementTiles(tileMetadata,null);
assert.equal(tileRequests.length,2);
assert.ok(tileRequests[1].url.includes('network=svv-met'));
tileRequests[0].handlers.tileerror();
assert.equal(tileErrors.length,0,'Old layers must not report errors');
tileRequests[1].handlers.tileerror();
assert.equal(tileErrors.length,1);
tileRequests[1].handlers.load();
assert.deepEqual(tileFinished.at(-1),[1,true]);
tileContext.removePlacementTiles();
assert.equal(tileContext.overlay,null);
tileContext.showUncertaintyTiles('uncertainty',{corridor:true,distance_m:500,bounds:[5,58,15,65]});
assert.equal(tileRequests.length,3);
assert.ok(tileRequests[2].url.includes('revision=7'));
assert.ok(tileRequests[2].url.includes('mode=uncertainty'));
assert.ok(tileRequests[2].url.includes('/api/map/tiles/'));
assert.equal(tileRequests[2].options.bounds,tileContext.state.bounds);
tileRequests[2].handlers.load();
tileContext.showUncertaintyTiles('uncertainty',{corridor:true,distance_m:500,bounds:[6,59,16,66]});
assert.equal(tileRequests.length,3,'Panning retains the uncertainty tile layer');
tileContext.showUncertaintyTiles('baseline',null);
assert.equal(tileRequests.length,4,'The baseline mode gets its own uncertainty tile layer');
tileContext.showUncertaintyTiles('reduction',null);
assert.equal(tileRequests.length,5,'Uncertainty reduction uses the sharp tile layer too');
assert.ok(tileRequests[4].url.includes('mode=reduction'));
let sliderPaints=0;
const localSlider={};
const sliderContext=vm.createContext({
  element:()=>localSlider,renderUncertaintyCutoff(){},clearPreview(){},clearTimeout(){},
  mode:'placement',state:{layers:['temperature:senorge30']},placementTileLayer:{},
  uncertaintyTileLayer:null,overlay:null,activeMapImage:null,pendingMapImage:null,
  repaintMapRaster:()=>sliderPaints++,setTimeout:()=>{throw Error('Local filtering must not schedule a server refresh');}
});
vm.runInContext(source.slice(source.indexOf('let rmseTimer=null;'),
  source.indexOf("document.querySelectorAll('[data-mode]').forEach(button=>button.onclick=")),sliderContext);
localSlider.oninput();localSlider.oninput();localSlider.oninput();
assert.equal(sliderPaints,3);
sliderContext.mode='uncertainty';sliderContext.overlay={rasterCanvas:{}};
localSlider.oninput();
sliderContext.mode='baseline';sliderContext.overlay=null;sliderContext.activeMapImage={rasterCanvas:{}};
localSlider.oninput();
sliderContext.activeMapImage=null;sliderContext.pendingMapImage={numeric:true};
localSlider.oninput();
assert.equal(sliderPaints,6,'Uncertainty, baseline and pending raster loads must all filter locally');
const legendRow={children:[],replaceChildren(){this.children=[];},append(item){this.children.push(item);},setAttribute(){}};
const legendRamp={hidden:false};
const legendQueries=[];
const legendContext=vm.createContext({
  state:{revision:1,layers:['temperature:senorge30']},mode:'uncertainty',
  element:id=>id==='legend-numbers'?legendRow:id==='ramp'?legendRamp:assert.fail(`Unexpected legend control: ${id}`),
  document:{createElement:()=>({style:{}})},shortNumber:String,number:Math.round,
  fetch:async url=>{
    legendQueries.push(url);
    const count=Number(new URL(url,'http://localhost').searchParams.get('count'));
    return {ok:true,json:async()=>({values:Array.from({length:count},(_,i)=>i===0?.1:i===count-1?10:1),unit:'°C',revision:1})};
  },
  error:message=>{throw Error(message);}
});
vm.runInContext(source.slice(source.indexOf('function usesEmptyReferenceLegend()'),source.indexOf('function updateMap()'))+
  source.slice(source.indexOf('let legendNumbersKey='),
  source.indexOf('function selectStation(')),legendContext);
(async()=>{
  await legendContext.refreshLegendNumbers();
  assert.equal(legendRow.hidden,false);
  assert.deepEqual(legendRow.children.map(item=>item.style.left),['0%','25%','50%','75%','100%']);
  assert.equal(legendRow.children[2].title,undefined);
  assert.equal(legendRow.children[0].textContent,'≤0.1');
  assert.equal(legendRow.children.at(-1).textContent,'≥10');
  assert.ok(legendQueries.at(-1).endsWith('&count=5'));
  legendContext.mode='placement';
  await legendContext.refreshLegendNumbers();
  assert.deepEqual(legendRow.children.map(item=>item.textContent),['0','1','10']);
  assert.deepEqual(legendRow.children.map(item=>item.style.left),['0%','50%','100%']);
  assert.ok(legendQueries.at(-1).endsWith('&count=3'));
  Object.assign(legendContext.state,{dataset:'temperature:direct_p95',layers:['temperature:direct_p95'],
    network:'empty',metric:'P95',comparison:'contribution',empty_reference_rmse:1});
  legendContext.mode='reduction';
  await legendContext.refreshLegendNumbers();
  assert.deepEqual(legendRow.children.map(item=>item.textContent),['≥1','0']);
  assert.ok(legendQueries.at(-1).endsWith('&count=2'));
  let resolve;
  legendContext.fetch=()=>new Promise(done=>{resolve=done;});
  legendContext.mode='uncertainty';
  const pending=legendContext.refreshLegendNumbers();
  legendContext.mode='climate';legendRamp.hidden=true;
  resolve({ok:true,json:async()=>({values:[1,2,3],unit:'m',revision:1})});
  await pending;
  assert.equal(legendRow.hidden,true,'Late numeric legends must not appear over categorical climate legends');
})().catch(error=>{console.error(error);process.exitCode=1;});
const roadCategoryBox={children:[],replaceChildren(){this.children=[];},append(item){this.children.push(item);}};
const defaultRoadElements={'road-show':{},'road-score-4':{checked:false},'road-score-5':{checked:false},
  'road-categories':roadCategoryBox};
const roadDefaults=vm.createContext({
  roadOptions:null,roadStatus(){},scheduleRoads(){},changeRoadFilters(){},
  element:id=>defaultRoadElements[id],ROAD_SCORE_COLOURS:['a','b','c','d','e'],
  document:{createElement:()=>({children:[],style:{},append(...items){this.children.push(...items);}}),createTextNode:String},
  fetch:async()=>({ok:true,json:async()=>({available:true,score_network:true,
    categories:[{code:'E',label:'Europaveg'},{code:'R',label:'Riksveg'},{code:'F',label:'Fylkesveg'}],
    classes:['1','2','3','4','5','U']})})
});
vm.runInContext(source.slice(source.indexOf('async function initializeRoads()'),
  source.indexOf("element('road-show').onchange=")),roadDefaults);
(async()=>{
  await roadDefaults.initializeRoads();
  const selected=roadCategoryBox.children.map(label=>label.children[0]).filter(input=>input.checked).map(input=>input.value);
  assert.deepEqual(selected,['E','R','F']);
  for(const id of ['road-score-4','road-score-5']){
    assert.equal(defaultRoadElements[id].checked,false);
    assert.equal(defaultRoadElements[id].disabled,false);
  }
})().catch(error=>{console.error(error);process.exitCode=1;});
const networkSelection=vm.createContext({state:{network:'frost',pross_only:true,include_met:true}});
vm.runInContext(source.slice(source.indexOf('const selectedNetwork ='),
  source.indexOf('\n',source.indexOf('const selectedNetwork ='))),networkSelection);
assert.equal(vm.runInContext('selectedNetwork()',networkSelection),'svv_met');
networkSelection.state.include_met=false;
assert.equal(vm.runInContext('selectedNetwork()',networkSelection),'svv');
networkSelection.state.pross_only=false;
assert.equal(vm.runInContext('selectedNetwork()',networkSelection),'frost');
vm.runInContext(source.slice(source.indexOf('const selectedPlacementNetwork ='),
  source.indexOf('\n',source.indexOf('const selectedPlacementNetwork ='))),networkSelection);
for(const [network,pross_only,include_met,expected] of [
  ['frost',false,false,'all'],['frost',true,false,'svv'],
  ['frost',true,true,'svv-met'],['empty',false,false,'empty']
]){
  networkSelection.state={network,pross_only,include_met};
  assert.equal(vm.runInContext('selectedPlacementNetwork()',networkSelection),expected);
}
const mapClickHandlers={},mapClickActions=[];
const clickContext=vm.createContext({
  mode:'placement',map:{on:(name,handler)=>{mapClickHandlers[name]=handler;}},
  element:()=>({checked:false}),
  promptNewStation:point=>mapClickActions.push(['add',point]),
  inspectKuldehull:point=>mapClickActions.push(['basin',point]),
  inspectBestStation:()=>{},inspectCoverage:()=>{},inspectTerrain:()=>{},inspectClimate:()=>{}
});
const mapClickStart=source.indexOf("map.on('click',async event=>{",source.indexOf("element('map').addEventListener('mouseleave'"));
vm.runInContext(source.slice(mapClickStart,source.indexOf("element('help-open').onclick=")),clickContext);
const clickedPoint={lat:60,lng:10};
mapClickHandlers.click({latlng:clickedPoint,originalEvent:{}});
clickContext.mode='uncertainty';
mapClickHandlers.click({latlng:clickedPoint,originalEvent:{}});
mapClickHandlers.click({latlng:clickedPoint,originalEvent:{ctrlKey:true}});
assert.deepEqual(mapClickActions.map(item=>item[0]),['add','add','basin']);
async function checkCoverageClickFallback(){
  for(const [mode,found,expected] of [
    ['climate',false,'climate'],['terrain:elevation',false,'terrain'],
    ['uncertainty',false,'add'],['placement',false,'add'],['climate',true,null]
  ]){
    const handlers={},actions=[];
    let activePopup=null;
    const context=vm.createContext({
      mode,busy:false,inspectionSerial:0,URLSearchParams,
      element:()=>({checked:true}),coverageClasses:()=>[5],
      document:{createElement:()=>({textContent:''})},
      fetch:async()=>({ok:true,json:async()=>({found,label:'Kandidatområde'})}),
      map:{
        on:(name,handler)=>{handlers[name]=handler;},
        hasLayer:popup=>activePopup===popup,
        closePopup:()=>{activePopup=null;handlers.popupclose();}
      },
      L:{popup:()=>({
        setLatLng(){return this;},setContent(content){this.content=content;return this;},
        openOn(){activePopup=this;return this;}
      })},
      inspectClimate:()=>actions.push('climate'),inspectTerrain:()=>actions.push('terrain'),
      promptNewStation:()=>actions.push('add'),inspectKuldehull:()=>actions.push('basin'),
      inspectBestStation:()=>actions.push('best')
    });
    vm.runInContext(source.slice(source.indexOf('async function inspectCoverage('),
      source.indexOf('async function initializeCoverage(')),context);
    vm.runInContext(source.slice(mapClickStart,source.indexOf("element('help-open').onclick=")),context);
    await handlers.click({latlng:clickedPoint,originalEvent:{}});
    assert.deepEqual(actions,expected?[expected]:[],`${mode}, coverage found=${found}`);
    if(found)assert.equal(activePopup.content.textContent,'Kandidatområde');
    else assert.equal(activePopup,null,'Empty coverage popups must close before the normal action');
    actions.length=0;
    await handlers.click({latlng:clickedPoint,originalEvent:{ctrlKey:true}});
    await handlers.click({latlng:clickedPoint,originalEvent:{altKey:true}});
    assert.deepEqual(actions,['basin','best']);
    if(mode==='climate'){
      let roadClick;
      context.L.geoJSON=(data,options)=>{
        options.onEachFeature({id:1},{on:(name,handler)=>{roadClick=handler;}});
        return {addTo(){return this;}};
      };
      context.L.DomEvent={stopPropagation(){}};
      context.roadRenderer={};context.roadStyle=()=>({});context.clearPreview=()=>{};
      context.showRoadPopup=()=>actions.push('road');
      vm.runInContext(source.slice(source.indexOf('const roadLines=L.geoJSON('),
        source.indexOf('function error(message)')),context);
      actions.length=0;
      await roadClick({latlng:clickedPoint,originalEvent:{}});
      assert.deepEqual(actions,found?[]:['road'],'Road clicks must retain their normal fallback');
    }
  }
}
checkCoverageClickFallback().catch(error=>{console.error(error);process.exitCode=1;});
assert.ok(template.includes('value="svv_met">Vegværstasjoner + MET-stasjoner'));
assert.equal(template.includes('id="empty-reference"'),false);
assert.equal(source.includes("element('empty-reference')"),false);
assert.equal(source.includes("action:'empty_reference'"),false);
class SensorFormElement{
  constructor(tag){this.tag=tag;this.children=[];this.style={};}
  append(...items){this.children.push(...items);}
  querySelectorAll(selector){
    const inputs=[];
    const visit=item=>{
      if(typeof item!=='object')return;
      if(item.tag==='input'&&(selector==='input'||item.checked))inputs.push(item);
      for(const child of item.children||[])visit(child);
    };
    visit(this);return inputs;
  }
}
let sensorForm;
const sensorActions=[],sensorErrors=[];
const sensorChoices=['temperature','wind','precipitation','snow','humidity','road_temperature'];
const sensorContext=vm.createContext({
  document:{createElement:tag=>new SensorFormElement(tag)},
  state:{layers:['temperature:senorge30'],layer_options:sensorChoices.map(phenomenon=>({
    phenomenon,dataset:phenomenon==='temperature'?'temperature:senorge30':`${phenomenon}:class`,
    available:phenomenon==='temperature'
  }))},
  SENSORS:{},sensorDot:()=>new SensorFormElement('span'),
  map:{closePopup:()=>{}},requestState:action=>sensorActions.push(action),error:message=>sensorErrors.push(message),
  L:{popup:()=>({setLatLng(){return this;},setContent(form){sensorForm=form;return this;},openOn(){return this;}})}
});
vm.runInContext(source.slice(source.indexOf('function promptNewStation('),
  source.indexOf("element('comparison-baseline').onclick=")),sensorContext);
sensorContext.promptNewStation({lat:63,lng:12});
assert.deepEqual(sensorForm.querySelectorAll('input').map(input=>input.value),sensorChoices);
assert.deepEqual(sensorForm.querySelectorAll('input:checked').map(input=>input.value),['temperature']);
for(const input of sensorForm.querySelectorAll('input'))input.checked=['temperature','wind','snow'].includes(input.value);
sensorForm.onsubmit({preventDefault(){}});
assert.deepEqual(Array.from(sensorActions[0].sensors),['temperature','wind','snow']);
assert.equal(sensorActions[0].action,'add');
sensorContext.promptNewStation({lat:63,lng:12},{id:7,sensors:['wind','snow']});
assert.deepEqual(sensorForm.querySelectorAll('input:checked').map(input=>input.value),['wind','snow']);
sensorForm.onsubmit({preventDefault(){}});
assert.deepEqual(Array.from(sensorActions[1].sensors),['wind','snow']);
assert.equal(sensorActions[1].action,'sensors');
assert.equal(sensorActions[1].id,7);
for(const input of sensorForm.querySelectorAll('input'))input.checked=false;
sensorForm.onsubmit({preventDefault(){}});
assert.deepEqual(sensorErrors,['Velg minst én sensor']);
assert.equal(sensorActions.length,2);
let stationPopup;
const stationActions=[];
let stationPopupClosed=0;
const stationEdits=[];
const stationContext=vm.createContext({
  document:{createElement:tag=>new SensorFormElement(tag)},
  state:{network:'frost',frost_ready:true,stations:[],layers:['temperature:senorge30']},
  element:()=>({checked:false}),map:{closePopup:()=>{stationPopupClosed++;}},
  requestState:action=>stationActions.push(action),selectStation:()=>{},
  promptNewStation:(latlng,station)=>stationEdits.push({latlng,station}),
  L:{popup:()=>({setLatLng(){return this;},setContent(box){stationPopup=box;return this;},openOn(){return this;}})}
});
vm.runInContext(source.slice(source.indexOf('const SENSORS='),
  source.indexOf('function promptNewStation(')),stationContext);
const originalStation={kind:'observation',station_id:'SN3',lon:12,lat:63,sensors:['snow'],layers:[]};
const popupButtons=()=>{const result=[];const visit=node=>{if(node.tag==='button')result.push(node);
  for(const child of node.children||[])if(typeof child==='object')visit(child);};visit(stationPopup);return result;};
stationContext.openStationPopup({lat:63,lng:12},originalStation);
assert.ok(popupButtons().some(button=>button.textContent==='Legg til temperatursensor'));
stationContext.state.stations=[{id:7,lon:12,lat:63,sensors:['temperature','wind']}];
stationContext.openStationPopup({lat:63,lng:12},originalStation);
const removeTemperature=popupButtons().find(button=>button.textContent==='Fjern temperatursensor');
assert.ok(removeTemperature);
removeTemperature.onclick();
assert.equal(stationActions[0].action,'remove_sensor');
assert.equal(stationActions[0].station_id,'SN3');
assert.equal(stationActions[0].sensor,'temperature');
stationContext.state.stations=[];
stationContext.openStationPopup({lat:63,lng:12},{...originalStation,sensors:['temperature']});
assert.equal(popupButtons().some(button=>button.textContent==='Fjern temperatursensor'),false);
stationContext.state.stations=[{id:7,lon:12,lat:63,sensors:['temperature']}];
stationContext.openStationPopup({lat:63,lng:12},{...originalStation,removed:true});
assert.equal(popupButtons().some(button=>button.textContent==='Fjern temperatursensor'),false);
const manualStation={kind:'manual',id:9,station_id:'Ny stasjon 2',lon:13,lat:64,sensors:['temperature']};
stationContext.openStationPopup({lat:64,lng:13},manualStation);
assert.deepEqual(popupButtons().map(button=>button.textContent),['Endre sensorer','Fjern målepunkt']);
popupButtons()[0].onclick();
assert.equal(stationEdits[0].station,manualStation);
const closedBeforeRemove=stationPopupClosed;
popupButtons()[1].onclick();
assert.equal(stationPopupClosed,closedBeforeRemove+1);
assert.equal(stationActions.at(-1).action,'remove');
assert.equal(stationActions.at(-1).id,9);

async function checkUpgradedStationSelection(){
  const actions=[],popups=[];
  const proposal={id:7,lon:12,lat:63,sensors:['temperature','wind']};
  const context=vm.createContext({
    state:{layers:['temperature:senorge30'],stations:[proposal],fixed_stations:[originalStation]},
    requestState:async action=>{actions.push(action);return true;},
    openStationPopup:(latlng,station)=>popups.push(station)
  });
  vm.runInContext(source.slice(source.indexOf('async function chooseFrostStation('),
    source.indexOf('const SENSORS=')),context);
  vm.runInContext(source.slice(source.indexOf('function sensorProposal('),
    source.indexOf('function sensorPie(')),context);
  await context.chooseFrostStation(originalStation,{lat:63,lng:12});
  assert.equal(actions[0].kind,'manual');
  assert.equal(actions[0].index,7);
  assert.equal(popups[0],originalStation,'Keep the observation popup and its sensor controls');
  context.state.layers=['snow:class'];
  await context.chooseFrostStation(originalStation,{lat:63,lng:12});
  assert.equal(actions[1].kind,'frost','An inactive proposal must not replace native sensor selection');
  assert.equal(actions[1].station_id,'SN3');
  context.state.layers=['temperature:senorge30','snow:class'];
  await context.chooseFrostStation(originalStation,{lat:63,lng:12});
  assert.equal(actions[2].kind,'manual');
  context.state.stations=[];
  await context.chooseFrostStation(originalStation,{lat:63,lng:12});
  assert.equal(actions[3].kind,'frost');
  assert.equal(actions[3].station_id,'SN3');
  context.state.stations=[proposal];
  await context.chooseFrostStation({...originalStation,removed:true},{lat:63,lng:12});
  assert.equal(actions[4].kind,'frost','Removed stations retain their restore comparison');
}
checkUpgradedStationSelection().catch(error=>{console.error(error);process.exitCode=1;});
const helpElements={
  'help-open':{},
  'help-close':{},
  'help-dialog':{open:false,showModal(){this.open=true;},close(){this.open=false;}}
};
vm.runInNewContext(source.slice(source.indexOf("element('help-open').onclick="),
  source.indexOf("element('station-visibility').onclick=")),{
  element:id=>{
    assert.ok(template.includes(`id="${id}"`),`Help references missing element: ${id}`);
    return helpElements[id];
  }
});
helpElements['help-open'].onclick();
assert.equal(helpElements['help-dialog'].open,true);
helpElements['help-dialog'].onclick({target:{}});
assert.equal(helpElements['help-dialog'].open,true,'Reading help must not close the dialog');
helpElements['help-close'].onclick();
assert.equal(helpElements['help-dialog'].open,false);
helpElements['help-open'].onclick();
helpElements['help-dialog'].onclick({target:helpElements['help-dialog']});
assert.equal(helpElements['help-dialog'].open,false);
helpElements['help-open'].onclick();
helpElements['help-dialog'].onkeydown({key:'Escape',preventDefault(){},stopPropagation(){}});
assert.equal(helpElements['help-dialog'].open,false);
for(const id of ['comparison-selected','comparison-panel','selected-station','selected-description',
  'frost-edit-status','remove-frost','restore-frost','road-ranking']){
  assert.equal(template.includes(`id="${id}"`),false,`${id} should not be displayed`);
  assert.equal(source.includes(`element('${id}')`),false,`${id} should not be required by the app`);
}
const comparisonButton={};
const comparisonActions=[];
const comparisonContext={element:id=>{assert.equal(id,'comparison-baseline');return comparisonButton;},
  state:{comparison_scope:'selected'},requestState:action=>comparisonActions.push(action)};
vm.runInNewContext(source.slice(source.indexOf("element('comparison-baseline').onclick="),
  source.indexOf("element('map').addEventListener('mouseleave'")),comparisonContext);
comparisonButton.onclick();
assert.equal(comparisonActions[0].value,'baseline');
comparisonContext.state.comparison_scope='baseline';
comparisonButton.onclick();
assert.equal(comparisonActions[1].value,'selected');
const roadDownload={};
const roadMessages=[];
const roadSummaryContext=vm.createContext({
  roadResult:{features:[{}],summary:{truncated:true}},roadOptions:{available:true},
  element:id=>{assert.equal(id,'road-download');return roadDownload;},
  roadStatus:message=>roadMessages.push(message)
});
vm.runInContext(source.slice(source.indexOf('function renderRoadSummary()'),
  source.indexOf('function roadStatus(')),roadSummaryContext);
roadSummaryContext.renderRoadSummary();
assert.equal(roadDownload.disabled,false);
assert.match(roadMessages.pop(),/Zoom inn/);
roadSummaryContext.roadResult={features:[],summary:{}};
roadSummaryContext.renderRoadSummary();
assert.equal(roadDownload.disabled,true);
const startupCalls=[];
const coverageToggle={};
vm.runInNewContext(source.slice(source.indexOf("element('coverage-show').onchange="),
  source.indexOf('function getRoadSettings(')),{
  element:id=>{
    assert.ok(template.includes(`id="${id}"`),`Startup references missing element: ${id}`);
    return id==='coverage-show'?coverageToggle:null;
  },
  scheduleCoverage:()=>{},
  initializeCoverage:()=>startupCalls.push('coverage'),
  requestState:()=>{startupCalls.push('state');return Promise.resolve();},
  state:null
});
assert.deepEqual(startupCalls,['coverage','state']);
assert.equal(typeof coverageToggle.onchange,'function');
const backgroundControls={
  'road-show':{checked:true},'road-score-4':{checked:false},'road-score-5':{checked:false},
  'road-reference':{value:'Rv3'},
  'road-categories':{querySelectorAll:selector=>selector==='input:checked'?categoryInputs.filter(i=>i.checked):categoryInputs}
};
const categoryInputs=[{value:'E',checked:true},{value:'R',checked:true},{value:'K',checked:true}];
let blackRoadsVisible=false,scoreRequests=0,clearedRoads=0;
const blackRoadUrls=[];
const blackRoads={setUrl:url=>blackRoadUrls.push(url),addTo:()=>{blackRoadsVisible=true;}};
const backgroundContext=vm.createContext({
  element:id=>backgroundControls[id],backgroundRoads:blackRoads,backgroundRoadKey:null,backgroundRoadError:false,
  roadOptions:{available:true,score_network:true,classes:['4','5']},state:{counties:['31']},URLSearchParams,
  map:{hasLayer:()=>blackRoadsVisible,removeLayer:()=>{blackRoadsVisible=false;}},
  clearTimeout:()=>{},setTimeout:()=>{scoreRequests++;return 1;},roadTimer:null,roadSerial:0,
  roadPending:false,roadController:null,roadResult:null,roadRequestKey:null,
  roadLines:{clearLayers:()=>clearedRoads++},roadCasing:{clearLayers:()=>clearedRoads++},
  renderRoadSummary:()=>{},refreshRoads:()=>{}
});
vm.runInContext(source.slice(source.indexOf('function selectedRoadScoreClasses()'),source.indexOf('function getRoadSettings('))+
  source.slice(source.indexOf('function scheduleRoads()'),
  source.indexOf('async function refreshRoads()')),backgroundContext);
backgroundContext.scheduleRoads();
assert.equal(blackRoadsVisible,true);
assert.equal(scoreRequests,0);
assert.equal(categoryInputs[2].disabled,true);
const blackQuery=new URL(blackRoadUrls[0],'http://localhost').searchParams;
assert.deepEqual(JSON.parse(blackQuery.get('categories')),['E','R']);
assert.equal(blackQuery.get('reference'),'Rv3');
assert.deepEqual(JSON.parse(blackQuery.get('counties')),['31']);
backgroundContext.scheduleRoads();
assert.equal(blackRoadUrls.length,1,'Panning must reuse the same filtered tile URL');
backgroundControls['road-score-4'].checked=true;
backgroundContext.scheduleRoads();
assert.equal(blackRoadsVisible,true);
assert.equal(scoreRequests,1);
assert.equal(categoryInputs[2].disabled,false);
backgroundControls['road-show'].checked=false;
backgroundContext.scheduleRoads();
assert.equal(blackRoadsVisible,false);
assert.equal(scoreRequests,2);
backgroundControls['road-score-4'].checked=false;
backgroundContext.scheduleRoads();
assert.equal(blackRoadsVisible,false);
assert.equal(scoreRequests,2);
assert.ok(clearedRoads>=4);
backgroundControls['road-show'].checked=true;
backgroundContext.scheduleRoads();
assert.equal(blackRoadsVisible,true);
const roadEvents={};
const moveHandlers={};
const roadContext={
  element:id=>template.includes(`id="${id}"`)?(roadEvents[id]??={addEventListener:()=>{}}):null,
  map:{on:(event,handler)=>{moveHandlers[event]=handler;}},
  scheduleRoads:()=>roadCalls.push('roads'),
  scheduleKuldehull:()=>roadCalls.push('kuldehull'),
  scheduleCoverage:()=>roadCalls.push('coverage'),
  changeRoadFilters:()=>{},
  updateMap:()=>roadCalls.push('map'),
  state:{}
};
const roadCalls=[];
vm.runInNewContext(source.slice(source.indexOf("element('road-show').onchange="),
  source.indexOf('async function loadPlacementOptions()')),roadContext);
assert.equal(typeof moveHandlers.moveend,'function');
moveHandlers.moveend();
assert.deepEqual(roadCalls,['roads','kuldehull','coverage','map']);

const roadControls={
  'road-show':{checked:false},
  'road-score-4':{checked:true},'road-score-5':{checked:true},
  'road-corridor':{checked:true},
  'road-distance':{value:'2000'},
  'road-categories':{querySelectorAll:()=>[{value:'E'}]},
  'road-reference':{value:'E6'}
};
const roadSettingsContext=vm.createContext({
  element:id=>roadControls[id],
  roadOptions:{available:true,score_network:true,categories:[{code:'E'},{code:'R'}],classes:['1','2','4','5']},
  map:{getBounds:()=>({getWest:()=>5,getSouth:()=>58,getEast:()=>30,getNorth:()=>71}),getZoom:()=>8}
});
vm.runInContext(source.slice(source.indexOf('function selectedRoadScoreClasses('),
  source.indexOf('function displayColourLimits(')),roadSettingsContext);
assert.deepEqual(Array.from(roadSettingsContext.getRoadSettings().classes),['4','5']);
assert.deepEqual(Array.from(roadSettingsContext.getRoadSettings().categories),['E']);
roadControls['road-score-5'].checked=false;
assert.deepEqual(Array.from(roadSettingsContext.getRoadSettings().classes),['4']);
roadControls['road-score-5'].checked=true;
roadControls['road-score-4'].checked=false;
assert.deepEqual(Array.from(roadSettingsContext.getRoadSettings().classes),['5']);
roadControls['road-score-5'].checked=false;
assert.equal(roadSettingsContext.getRoadSettings(),null);
const weatherRoads=JSON.parse(JSON.stringify(roadSettingsContext.getWeatherRoadSettings()));
assert.deepEqual(weatherRoads.categories,['E','R']);
assert.deepEqual(weatherRoads.classes,['1','2','4','5']);
assert.equal(weatherRoads.reference,'');
assert.equal(weatherRoads.distance_m,2000);
roadControls['road-corridor'].checked=false;
assert.equal(roadSettingsContext.getWeatherRoadSettings(),null);
assert.equal(roadSettingsContext.getWeatherRoadSettings(true).distance_m,2000);
assert.deepEqual(Array.from(roadSettingsContext.getWeatherRoadSettings(true).categories),['E','R']);
const roadStyleContext=vm.createContext({});
vm.runInContext(source.slice(source.indexOf('function displayRoadScoreClass('),
  source.indexOf('function emptyReductionLegendTitle(')),roadStyleContext);
assert.equal(roadStyleContext.displayRoadScoreClass(5),1);
assert.equal(roadStyleContext.displayRoadScoreClass('4'),2);
assert.equal(roadStyleContext.displayRoadScoreClass(3),3);
vm.runInContext(source.match(/^const ROAD_SCORE_COLOURS=.*$/m)[0]+'\n'+source.slice(source.indexOf('function roadStyle('),
  source.indexOf('function scheduleRoads()')),roadStyleContext);
assert.notEqual(roadStyleContext.roadStyle({properties:{VS_Total_klasse:4}}).color,
  roadStyleContext.roadStyle({properties:{VS_Total_klasse:5}}).color);

const pendingTimers=new Map();
let nextTimer=0;
const coverageElements={
  'coverage-classes':{querySelectorAll:()=>[{value:'5'}]},
  'coverage-show':{checked:true},
  'coverage-status':{textContent:''}
};
const coverageContext=vm.createContext({
  element:id=>coverageElements[id],
  state:{counties:['03']},
  getKuldehullSettings:()=>({bounds:[5,60,15,65],width:800,height:600}),
  setTimeout:callback=>{const id=++nextTimer;pendingTimers.set(id,callback);return id;},
  clearTimeout:id=>pendingTimers.delete(id),
  AbortController,URLSearchParams,
  map:{removeLayer:()=>{}}
});
vm.runInContext(source.slice(source.indexOf('let coverageOptions='),
  source.indexOf('async function inspectCoverage(')),coverageContext);
vm.runInContext('coverageOptions={available:true};scheduleCoverage();scheduleCoverage();',coverageContext);
assert.equal(pendingTimers.size,1,'Repeated scheduling must not cancel the pending coverage request');
coverageElements['coverage-show'].checked=false;
vm.runInContext('scheduleCoverage()',coverageContext);
assert.equal(pendingTimers.size,0,'Disabling coverage must cancel a pending request');
const input={id:'opacity',type:'range',value:'40',checked:undefined,parentElement:{closest:()=>({id:'panel'})}};
const details={id:'phenomena-picker',open:true};
const elements={
  'kuldehull-show':{checked:false},
  'kuldehull-status':{textContent:''},
  'kuldehull-depth':{value:'20'},
  'kuldehull-only':{checked:false},
  'road-show':{checked:false},
  'road-scores':{checked:false},
  'road-corridor':{checked:false},
  'coverage-classes':{querySelectorAll:()=>[{value:'5'}]},
  'coverage-show':{checked:false},
  'opacity':{value:'40'}
};
class ImageOverlay{
  options={opacity:.6,pane:'overlayPane'};
  getElement(){return {complete:true,naturalWidth:1,src:'data:image/png;base64,AAAA'};}
  getBounds(){return {getSouth:()=>60,getWest:()=>5,getNorth:()=>65,getEast:()=>15};}
}
class Marker{
  options={icon:{options:{html:'',className:'frost-pin',iconSize:[14,14],iconAnchor:[7,7]}},pane:'stations'};
  getLatLng(){return {lat:63,lng:12};}
}
const downloads=[],failures=[];
const coverageImage=new ImageOverlay();
const context=vm.createContext({
  console,FileReader:class{},setTimeout,URL,Blob,Response,CompressionStream,btoa,
  document:{
    title:'Test',
    querySelectorAll:selector=>selector==='aside input,aside select'?[input]:
      selector==='aside details[id]'?[details]:selector==='style'?[{textContent:'.frost-pin{border:2px solid purple}'}]:[],
    querySelector:()=>({outerHTML:'<div class="legend">Temperature</div>'})
  },
  element:id=>elements[id],controls:()=>{},error:message=>failures.push(message),
  commitPendingCountySelection:async()=>true,updateMap:()=>{},
  state:{revision:7,dataset:'snow:class',bounds:[[60,5],[65,15]],
    station_owners:{available:true,updated:'2026-11-11',stations:1,unknown:0,owners:[{name:'OWNER',count:1}]},
    fixed_stations:[{lon:12,lat:63,station_id:'SN1',name:'</script><script>bad()</script>',sensors:['snow'],owners:['OWNER']}],stations:[]},
  roadOptions:{available:true},
  busy:false,mode:'uncertainty',stationsVisible:true,
  fixedMarkers:{eachLayer:()=>{}},manualMarkers:{eachLayer:()=>{}},
  buildCalculationPoints:()=>{},getKuldehullSettings:()=>({bounds:[5,60,15,65],width:800,height:600,depth:20}),
  overlayOpacity:()=>.6,URLSearchParams,
  coverageImage,
  map:{getCenter:()=>({lat:63,lng:12}),getZoom:()=>6,hasLayer:()=>true,
    eachLayer:callback=>[new ImageOverlay(),coverageImage,new Marker()].forEach(callback)},
  L:{ImageOverlay,Marker,Path:class{},layerGroup:()=>({eachLayer:()=>{}})},loadingTasks:new Map(),roadBusy:false,roadPending:false,kuldehullOverlay:null,
  sensorProposal:()=>null,rmseMinimum:()=>0,SENSORS:{snow:['Snow','#aaa']},
  nativeFetch:async()=>({ok:true,blob:async()=>new Blob()}),
  fetch:async()=>({ok:true,json:async()=>({format:'vaernett-scenario',version:1})})
});
vm.runInContext(source.slice(0,source.indexOf('const element =')),context);
vm.runInContext("coverageImages.add(coverageImage);coverageOptions={classes:[{value:5,label:'Svært interessant kandidatområde',color:'#d73027'}]};",context);
context.getRoadSettings=()=>({categories:['E'],classes:['1']});
context.getWeatherRoadSettings=includeHidden=>includeHidden?
  {categories:['E','R'],classes:['1','2'],reference:'',corridor:true,distance_m:500}:null;
const full={revision:7,roads:{count:2,chunks:[]},basins:{count:3,chunks:[]},coverage:{count:1,chunks:[]},
  road_overview:{type:'FeatureCollection',features:[]},
  placement:{url:'data:image/png;base64,AAAA',roads_url:'data:image/png;base64,BBBB',bounds:[[60,5],[65,15]],station_network:'all'},
  weather:['counties','roads'].map(id=>({id,label:id==='roads'?'Usikkerhet langs veger':'Usikkerhetskart',unit:'°C',
    distance_m:id==='roads'?500:null,
    minimum:.01,maximum:10,bounds:[[60,5],[65,15]],url:'data:image/png;base64,AAAA',
    baseline:'data:image/png;base64,AAAA',reduction:'data:image/png;base64,AAAA'}))};
let exportedCoverageClasses,exportedWeatherRoads,exportedRoadScores,exportedPlacement;
let exportedRoadsVisible;
context.fetch=async (url,options)=>{
  if(url==='/api/snapshot-data'){
    const body=JSON.parse(options.body);
    exportedCoverageClasses=body.coverage_classes;
    exportedWeatherRoads=body.weather_roads;
    exportedRoadScores=body.road_scores;
    exportedRoadsVisible=body.roads_visible;
    exportedPlacement=body.include_placement;
  }
  return {ok:true,json:async()=>url==='/api/snapshot-data'?full:
  url==='/api/scenario'?{format:'vaernett-scenario',version:1}:{zones:[]}};
};
context.downloadSavedFile=(content,type,name)=>downloads.push({content,type,name});
context.imageData=async()=> 'data:image/png;base64,AAAA';
(async()=>{
  const view=context.savedView();
  context.validateSavedView(JSON.parse(JSON.stringify(view)));
  assert.equal(view.controls[0].checked,false);
  assert.throws(()=>context.validateSavedView({...view,center:[NaN,12]}));
  assert.throws(()=>context.validateSavedView({...view,mode:'unknown'}));
  context.validateSavedView({...view,mode:'terrain:local_relief'});
  await context.saveEditableScenario();
  assert.equal(JSON.parse(downloads.pop().content).view.controls[0].value,'40');
  await context.saveFrozenMap();
  assert.deepEqual(failures,[]);
  const html=downloads.pop().content;
  assert.equal(html.includes('/api/'),false);
  assert.equal(html.includes('<script>bad()'),false);
  const snapshot=JSON.parse(html.match(/id="snapshot">([\s\S]*?)<\/script>/)[1]);
  assert.equal(exportedRoadScores,false);
  assert.equal(exportedRoadsVisible,false);
  assert.equal(exportedPlacement,true);
  const preview=JSON.parse(require('node:zlib').gunzipSync(Buffer.from(snapshot.roadPreview,'base64')));
  assert.deepEqual(preview[0].geojson,full.road_overview);
  assert.equal(preview[0].pane,'background-roads');
  assert.equal(preview[0].style.color,'#000');
  assert.equal(snapshot.markers.length,1); // Empty-HTML station symbols must not disappear.
  assert.equal(snapshot.images,undefined);
  assert.equal(snapshot.weatherExtent,'counties');
  assert.equal(snapshot.opacity,'40');
  assert.equal(snapshot.coverageImages,undefined);
  assert.deepEqual(snapshot.coverageOverview.bounds,context.state.bounds);
  assert.equal(snapshot.full.coverage.count,1);
  assert.deepEqual(exportedCoverageClasses,[5]);
  assert.equal(exportedWeatherRoads.distance_m,500);
  assert.deepEqual(exportedWeatherRoads.categories,['E','R']);
  assert.equal(snapshot.coverageVisible,false);
  assert.equal(snapshot.coverageClasses[0].value,5);
  assert.ok(html.includes("overlays['Dekningsgrad']"));
  assert.ok(html.includes("box.className='legend climate-snapshot-legend'"));
  assert.equal(snapshot.full.roads.count,2);
  assert.equal(snapshot.full.basins.count,3);
  assert.equal(snapshot.full.weather.length,2);
  assert.equal(snapshot.baseline,undefined);
  assert.equal(snapshot.uncertainty,undefined);
  assert.equal(html.includes('Værfenomen'),false);
  assert.equal(html.includes('Lagret visning'),false);
  assert.ok(html.includes("'Reduksjon':reductionWeather"));
  assert.ok(html.includes("'Plasseringsgevinst':placementWeather"));
  assert.ok(html.includes('id="saved-road-only"'));
  assert.ok(html.includes('Vis usikkerhet langs veg'));
  assert.equal(html.includes('id="saved-weather"'),false);
  assert.equal(html.includes('Korridoravstand:'),false);
  assert.ok(html.includes('Viser usikkerhet inntil 500 m fra veg'));
  assert.ok(snapshot.basinOverview.url.startsWith('data:image/'));
  assert.deepEqual(snapshot.basinOverview.bounds,[[60,5],[65,15]]);
  assert.equal(snapshot.basinsVisible,false);
  assert.equal(snapshot.markers[0].sensors[0].color,'#aaa');
  assert.deepEqual(snapshot.markers[0].info.owners,['OWNER']);
  assert.equal(snapshot.stationOwners.owners[0].count,1);
  assert.equal(html.includes('id="saved-owners"'),false);
  assert.ok(snapshot.climate.url.startsWith('data:image/'));
  assert.equal(snapshot.mode,'uncertainty');
  assert.equal(snapshot.markers[0].title,'SN1 · </script><script>bad()</script>');
  const runtime=html.match(/<script>([\s\S]*?)<\/script>/)[1];
  new vm.Script(runtime);
  assert.equal(runtime.includes('Observasjoner fra'),false);
  assert.equal(runtime.includes('Observasjoner til'),false);
  assert.ok(runtime.includes('tileerror'));
  assert.ok(runtime.includes('World_Topo_Map'));
  assert.ok(runtime.includes("map.on('click'"));
  assert.equal(html.includes('id="toggle-climate"'),false);
  assert.ok(runtime.includes("L.control.layers({'Usikkerhet':weather,'Grunnett':baselineWeather,'Reduksjon':reductionWeather,'Plasseringsgevinst':placementWeather,'Vegklimasonekart':climate"));
  assert.ok(runtime.includes("'Bare bakgrunnskart':backgroundOnly"));
  assert.ok(runtime.includes("weatherLegendMarkup('Temperatur')"));
  assert.equal(runtime.includes("activeWeather.unit"),false);
  assert.ok(runtime.includes("const visible=[['coverage',coverage],['basins',basins],['roads',roads]]"));
  assert.ok(runtime.includes("renderer:kind==='roads'?backgroundRoadRenderer:detailRenderer"));
  assert.ok(runtime.includes('debug.addTo(stations)'));
  assert.ok(runtime.includes("new DecompressionStream('gzip')"));
  assert.ok(runtime.includes('intersects(feature.bbox,view)'));
  assert.ok(runtime.includes('bubblingMouseEvents:false'));
  assert.ok(runtime.includes("pane:'basins'"));
  assert.ok(runtime.includes("pane:'weather',interactive:false"));
  assert.ok(runtime.includes("['coverage',425]"));
  assert.ok(runtime.includes("interactive:false,pane:'coverage'"));
  assert.ok(runtime.includes("map.getPane('county-borders').style.pointerEvents='none'"));
  assert.equal(html.includes('data-weather-mode='),false);
  assert.equal(runtime.includes('coordinateRows'),false);
  assert.ok(html.includes('.leaflet-tile-pane{filter:none}'));
  context.mode='climate';
  context.state.bounds=[[60,5],[65,15]];
  context.state.layers=['snow:continuous'];
  await context.saveFrozenMap();
  const climateSnapshot=JSON.parse(downloads.pop().content.match(/id="snapshot">([\s\S]*?)<\/script>/)[1]);
  assert.equal(climateSnapshot.images,undefined);
  assert.ok(climateSnapshot.climate.url.startsWith('data:image/'));
  context.mode='reduction';
  elements['road-corridor'].checked=true;
  await context.saveFrozenMap();
  const reductionSnapshot=JSON.parse(downloads.pop().content.match(/id="snapshot">([\s\S]*?)<\/script>/)[1]);
  assert.equal(reductionSnapshot.mode,'reduction');
  assert.equal(reductionSnapshot.weatherExtent,'roads');
  assert.equal(reductionSnapshot.images,undefined);
  context.mode='placement';
  await context.saveFrozenMap();
  const placementSnapshot=JSON.parse(downloads.pop().content.match(/id="snapshot">([\s\S]*?)<\/script>/)[1]);
  assert.equal(placementSnapshot.mode,'placement');
  assert.equal(placementSnapshot.full.placement.station_network,'all');
  context.mode='uncertainty';
  const moved=context.state.fixed_stations[0];
  moved.svv=true;moved.original_position={lon:11,lat:62};
  const original=new Marker();original.getLatLng=()=>({lat:62,lng:11});
  original.options={...original.options,savedStation:moved,originalStationPosition:true};
  context.map.eachLayer=callback=>[new ImageOverlay(),new Marker(),original].forEach(callback);
  await context.saveFrozenMap();
  const movedHTML=downloads.pop().content;
  const movedSnapshot=JSON.parse(movedHTML.match(/id="snapshot">([\s\S]*?)<\/script>/)[1]);
  assert.equal(movedSnapshot.markers.length,2);
  assert.ok(movedSnapshot.markers.every(marker=>marker.info.svv===true));
  assert.equal(movedSnapshot.markers[1].original,true);
  assert.equal(movedSnapshot.markers[1].info.station_id,moved.station_id);
  if(process.env.SNAPSHOT_BROWSER_FIXTURE){
    const zlib=require('node:zlib');
    const png='data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIHWP4z8DwHwAFgAI/ScLbtAAAAABJRU5ErkJggg==';
    const feature={type:'Feature',bbox:[12.02,63.02,12.04,63.04],properties:{lake_id:7,max_depth_m:30,area_m2:10000},
      geometry:{type:'Polygon',coordinates:[[[12.02,63.02],[12.04,63.02],[12.04,63.04],[12.02,63.04],[12.02,63.02]]]}};
    const chunk={bounds:feature.bbox,count:1,data:zlib.gzipSync(JSON.stringify({type:'FeatureCollection',features:[feature]})).toString('base64')};
    movedSnapshot.center=[63.03,12.03];movedSnapshot.zoom=13;movedSnapshot.basinsVisible=true;movedSnapshot.roadsVisible=true;
    movedSnapshot.climate.bounds=[[60,5],[65,15]];
    movedSnapshot.coverageOverview=snapshot.coverageOverview;
    movedSnapshot.coverageVisible=true;
    movedSnapshot.coverageClasses=snapshot.coverageClasses;
    movedSnapshot.full.coverage={count:1,chunks:[{...chunk,
      bounds:[12.06,63.02,12.08,63.04],
      data:zlib.gzipSync(JSON.stringify({type:'FeatureCollection',features:[{
        type:'Feature',bbox:[12.06,63.02,12.08,63.04],properties:{value:5},
        geometry:{type:'Polygon',coordinates:[[[12.06,63.02],[12.08,63.02],[12.08,63.04],[12.06,63.04],[12.06,63.02]]]}
      }]})).toString('base64')} ]};
    movedSnapshot.full.basins={count:1,chunks:[chunk]};
    movedSnapshot.full.roads={count:1,chunks:[{...chunk,data:zlib.gzipSync(JSON.stringify({type:'FeatureCollection',
      features:[{...feature,properties:{VS_Total:40,VS_Total_klasse:3},geometry:{type:'LineString',coordinates:[[12.02,63.03],[12.04,63.03]]}}]})).toString('base64')}]};
    movedSnapshot.vectors.push({pane:'county-borders',style:{color:'black',weight:8},geojson:{...feature,
      geometry:{type:'LineString',coordinates:[[12.03,63.02],[12.03,63.04]]}}});
    let fixture=movedHTML.replace(/id="snapshot">([\s\S]*?)<\/script>/,
      'id="snapshot">'+JSON.stringify(movedSnapshot).replace(/</g,'\\u003c')+'</script>');
    fixture=fixture.replaceAll('data:image/png;base64,AAAA',png).replace('</head>','<style>.ramp{height:10px}.ticks{display:flex;justify-content:space-between}</style></head>');
    fixture=fixture.replace("const map=L.map('map'","const map=window.testMap=L.map('map'");
    fs.writeFileSync(process.env.SNAPSHOT_BROWSER_FIXTURE,fixture);
  }
  context.roadBusy=true;
  await context.saveFrozenMap();
  assert.equal(downloads.length,0);
  assert.equal(failures.length,1);
  context.roadBusy=false;
  context.fetch=async()=>({ok:false,json:async()=>({error:'Climate unavailable'})});
  await context.saveFrozenMap();
  assert.equal(downloads.length,0);
  assert.equal(failures.at(-1),'Climate unavailable');
  context.fetch=async url=>({ok:url!=='/api/snapshot-data',
    json:async()=>url==='/api/snapshot-data'?{error:'Basins unavailable'}:{zones:[]}});
  await context.saveFrozenMap();
  assert.equal(downloads.length,0);
  assert.equal(failures.at(-1),'Basins unavailable');
  console.log('Scenario UI and frozen HTML tests passed');
})().catch(error=>{console.error(error);process.exitCode=1;});
