// Run with: node test_station_saving.js
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const source=fs.readFileSync('station_app.js','utf8');
const template=fs.readFileSync('station_app.html','utf8');
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
const full={revision:7,roads:{count:2,chunks:[]},basins:{count:3,chunks:[]},coverage:{count:1,chunks:[]},
  weather:['temperature','wind','precipitation','snow','humidity','road_temperature'].map(id=>({id,label:id,unit:'test',
    minimum:.01,maximum:10,bounds:[[60,5],[65,15]],url:'data:image/png;base64,AAAA',
    baseline:'data:image/png;base64,AAAA'}))};
let exportedCoverageClasses;
context.fetch=async (url,options)=>{
  if(url==='/api/snapshot-data')exportedCoverageClasses=JSON.parse(options.body).coverage_classes;
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
  assert.equal(snapshot.markers.length,1); // Empty-HTML station symbols must not disappear.
  assert.deepEqual(snapshot.images[0].bounds,[[60,5],[65,15]]);
  assert.equal(snapshot.images[0].opacity,.6);
  assert.equal(snapshot.images.filter(image=>image.basins).length,0);
  assert.equal(snapshot.images.length,1);
  assert.equal(snapshot.coverageImages,undefined);
  assert.deepEqual(snapshot.coverageOverview.bounds,context.state.bounds);
  assert.equal(snapshot.full.coverage.count,1);
  assert.deepEqual(exportedCoverageClasses,[5]);
  assert.equal(snapshot.coverageVisible,false);
  assert.equal(snapshot.coverageClasses[0].value,5);
  assert.ok(html.includes("overlays['Dekningsgrad']"));
  assert.ok(html.includes("box.className='legend climate-snapshot-legend'"));
  assert.equal(snapshot.full.roads.count,2);
  assert.equal(snapshot.full.basins.count,3);
  assert.equal(snapshot.full.weather.length,6);
  assert.ok(snapshot.baseline.url.startsWith('data:image/'));
  assert.ok(snapshot.uncertainty.url.startsWith('data:image/'));
  assert.ok(snapshot.weatherLegend.includes('Mindre usikkert'));
  assert.ok(snapshot.weatherLegend.includes('Mer usikkert'));
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
  assert.ok(snapshot.images[0].url.startsWith('data:image/'));
  const runtime=html.match(/<script>([\s\S]*?)<\/script>/)[1];
  new vm.Script(runtime);
  assert.ok(runtime.includes('tileerror'));
  assert.ok(runtime.includes('World_Topo_Map'));
  assert.ok(runtime.includes("map.on('click'"));
  assert.equal(html.includes('id="toggle-climate"'),false);
  assert.ok(runtime.includes("L.control.layers({'Usikkerhet':weather,'Grunnett':baselineWeather,'Klimasoner':climate"));
  assert.ok(runtime.includes("'Bare bakgrunnskart':backgroundOnly"));
  assert.ok(runtime.includes("const visible=[['coverage',coverage],['basins',basins],['roads',roads]]"));
  assert.ok(runtime.includes("renderer:detailRenderer"));
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
  assert.equal(climateSnapshot.images.length,1);
  assert.equal(climateSnapshot.images.filter(image=>image.basins).length,0);
  assert.ok(climateSnapshot.climate.url.startsWith('data:image/'));
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
