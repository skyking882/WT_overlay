'use strict';
(() => {
  const $ = id => document.getElementById(id);
  const canvas = $('map'), ctx = canvas.getContext('2d');
  const P = {catalog: null, state: null, scene: null, dirty: false, selected: 0, busy: false,
    view: '3d', pending: null, camera: {x: 0, y: 0, z: 2500, yaw: .45, elevation: .62, radius: 120000, orbit: 0}, received: 0,
    map: {x: 0, y: 0, span: 140000}, screen: [], trails: new Map(), events: new Map(), keys: new Set(),
    drag: null, lastTime: -1, lastStatus: '', formSignature: '', inputPending: false,
    mode: 'setup', replays: [], me: '', showAI: false, seeking: false, markerFile: null, meFile: null, lastFork: false,
    sel: [], ownId: null, missileOwner: new Map(), trackSig: {}, wrplMine: new Map(), wrplLoaded: new Set(), wrplSig: '', dragDepth: 0};
  const teams = ['蓝队', '橙队'], colors = ['#54bdff', '#ffaf64'];
  // Display name and country group per airframe; unknown ids fall back to the raw id.
  const names = {
    f_15c_golden_eagle:['F-15C 金鹰','美国'],f_15c_baz_msip:['F-15C Baz MSIP','以色列'],f_15c_msip2:['F-15C MSIP II','美国'],
    f_15e:['F-15E','美国'],f_15i_raam:['F-15I Ra\'am','以色列'],f_15j_kai:['F-15J 改','日本'],
    f_16a_block_15_adf:['F-16A ADF','美国'],f_16a_block_15_adf_italy:['F-16A ADF','意大利'],f_16a_block_15_ocu_thailand:['F-16A OCU','泰国'],
    f_16a_block_20_mlu:['F-16A MLU','中华民国'],f_16a_block_72v_china:['F-16V Block 72','中华民国'],f_16am_block_15_mlu_belgium:['F-16AM MLU','比利时'],
    f_16am_block_20_mlu_netherlands:['F-16AM MLU','荷兰'],f_16am_block_20_mlu_norway:['F-16AM MLU','挪威'],f_16c_block_40_barak_2:['F-16C Barak II','以色列'],
    f_16c_block_50:['F-16C Block 50','美国'],f_16c_block_52_aesa:['F-16C Block 52 AESA','美国'],f_16d_block_40_barak_2:['F-16D Barak II','以色列'],
    f_16i_sufa:['F-16I Sufa','以色列'],f_16xl:['F-16XL','美国'],f_2a:['F-2A','日本'],
    'f-4f_kws_lv':['F-4F ICE','德国'],f_4e_aup_greece:['F-4E AUP','希腊'],'f-5th_thailand':['F-5TH','泰国'],
    fa_18a_hug_2_raaf:['F/A-18A HUG 2','澳大利亚'],fa_18c_finland:['F/A-18C','芬兰'],fa_18c_late:['F/A-18C Late','美国'],
    fa_18c_late_switzerland:['F/A-18C Late','瑞士'],fa_18c_mlu_2_finland:['F/A-18C MLU 2','芬兰'],fa_18d_late_malaysia:['F/A-18D','马来西亚'],
    fa_18e_block_2:['F/A-18E Block II','美国'],fa_18f_block_2_raaf:['F/A-18F Block II','澳大利亚'],
    av_8b_plus:['AV-8B+','美国'],av_8b_plus_italy:['AV-8B+','意大利'],sea_harrier_fa2:['海鹞 FA2','英国'],tornado_f3_late:['狂风 F3 Late','英国'],
    ef_2000_aesa:['台风 AESA','德国'],ef_2000_block_10:['台风 Block 10','西班牙'],ef_2000_fgr4:['台风 FGR4','英国'],
    ef_2000_typhoon_aesa:['台风 ECRS','英国'],ef_2000a:['台风 EF-2000A','意大利'],ef_2000a_aesa:['台风 EF-2000A AESA','意大利'],
    saab_ja37di:['JA 37DI','瑞典'],saab_ja37di_f21:['JA 37DI F21','瑞典'],saab_jas39c:['JAS 39C','瑞典'],saab_jas39d:['JAS 39D','瑞典'],
    saab_jas39e:['JAS 39E','瑞典'],saab_jas39c_hungary:['JAS 39C','匈牙利'],saab_jas39c_south_africa:['JAS 39C','南非'],saab_jas39c_thailand:['JAS 39C','泰国'],
    kfir_c10_colombia:['幼狮 C10','以色列'],m_346fa:['M-346FA','意大利'],
    j_10a:['歼-10A','中国'],j_10c:['歼-10C','中国'],j_11a:['歼-11A','中国'],j_11b:['歼-11B','中国'],j_15t:['歼-15T','中国'],
    j_16:['歼-16','中国'],j_8f:['歼-8F','中国'],jf_17:['JF-17','中国'],jh_7a2:['歼轰-7A II','中国'],su_30mkk:['苏-30MKK','中国'],
    mig_29kr_9_41r:['米格-29KR','苏联/俄罗斯'],mig_29m_9_15:['米格-29M','苏联/俄罗斯'],mig_29n:['米格-29N','马来西亚'],
    mig_29smt_9_19:['米格-29SMT','苏联/俄罗斯'],mig_35:['米格-35','苏联/俄罗斯'],su_27sm:['苏-27SM','苏联/俄罗斯'],
    su_30mk2v_venezuela:['苏-30MK2','委内瑞拉'],su_30mkm:['苏-30MKM','马来西亚'],su_30sm:['苏-30SM','苏联/俄罗斯'],
    su_30sm2:['苏-30SM2','苏联/俄罗斯'],su_34:['苏-34','苏联/俄罗斯']};
  const missileNames = {us_aim_120a:'AIM-120A',us_aim_120b:'AIM-120B',us_aim_120c_5:'AIM-120C-5',us_aim_120d:'AIM-120D',
    su_r_77:'R-77',su_r_77_1:'R-77-1',su_rvv_ae:'RVV-AE',su_rvv_sd:'RVV-SD',cn_pl12:'霹雳-12',cn_pl12a:'霹雳-12A',cn_sd10a:'SD-10A',
    swd_rb99:'Rb 99',il_derby:'Derby',jp_aam4:'AAM-4',r_darter:'R-Darter'};
  const archetypes = {left:'左侧迂回',right:'右侧迂回',middle:'中路推进',crawler:'低空接敌',rusher:'快速突进'};
  const phases = {climb:'爬升',advance:'推进',evade:'防御',recommit:'再次接敌',round2:'第二轮',rush:'突进',
    home:'返航',suppress:'压制',crawl:'低空接敌',popup:'跃升',manual:'人工导航',pilot:'亲自驾驶',track:'WT 轨迹'};
  const controlModes = {track:'WT 轨迹',auto:'自主脚本',manual:'人工导航',pilot:'亲自驾驶'};
  const outcomes = {hit:'命中',hit_nokill:'命中未击落',miss:'脱靶',target_dead:'目标已先损失',unknown:'未知'};
  const reasons = {annihilation:'一方全灭',mutual_annihilation:'双方全灭',time_limit:'到达时间上限',stalemate:'僵局'};
  const fmt = (v, n=0) => v == null ? '—' : Number(v).toFixed(n);
  const title = id => names[id]?.[0] || id.replaceAll('_', ' ').toUpperCase();
  const families = [[/^f_15/,'F-15'],[/^f_16/,'F-16'],[/^fa_18/,'F/A-18'],[/^ef_2000/,'台风'],[/^saab/,'萨博'],
    [/^(j_|jf_|jh_|su_30mkk)/,'中国'],[/^mig/,'米格'],[/^su_/,'苏霍伊']];
  const family = id => families.find(([re])=>re.test(id))?.[1] || '其他';
  const missileName = id => id ? missileNames[id] || id.replace(/^(us|su|fr|uk|cn|il|ja|de)_/, '').replaceAll('_', '-').toUpperCase() : '无挂载';
  const clamp = (n, a, b) => Math.max(a, Math.min(b, n));
  const clone = o => JSON.parse(JSON.stringify(o));
  const dot = (a,b) => a.reduce((s,v,i) => s+v*b[i],0);
  const cross = (a,b) => [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
  const unit = a => { const n=Math.hypot(...a); return a.map(v=>v/n); };
  const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
  const short = name => name ? (name.length>14 ? name.slice(0,14)+'…' : name) : '';
  const baseMissile = id => typeof id==='string'&&id.endsWith('_default') ? id.slice(0,-8) : id;
  const clockText = t => String(Math.floor(t/60)).padStart(2,'0')+':'+fmt(t%60,2).padStart(5,'0');
  // Replay helpers: playback (status "playback") and forks (state.fork) share the WT replay side of the UI.
  const inReplay = () => !!P.state && (P.state.status==='playback' || !!P.state.fork);
  const roster = () => P.state?.replay?.roster || P.state?.fork?.roster || null;
  const nameOf = id => { const r=roster()?.[id]; return r?.player ? short(r.player) : '#'+id; };
  const mapHalf = () => P.state?.map_half_m || P.scene.map_half_m;
  // 本机: in playback the replay's 我的飞机; otherwise the aircraft set with 设为本机, the fork's own aircraft, or the
  // one being flown by keyboard.
  function ownId(){
    const s=P.state;if(!s)return null;const ps=s.planes||[];
    if(s.status==='playback'){const p=P.me?ps.find(p=>p.player===P.me):null;return p?p.id:null;}
    if(s.status==='setup')return null;
    if(P.ownId!=null&&ps.some(p=>p.id===P.ownId))return P.ownId;
    if(s.fork?.me!=null)return s.fork.me;
    const pilot=ps.find(p=>p.mode==='pilot'&&p.alive);return pilot?pilot.id:null;
  }
  function isMe(p){ return p!=null&&p.id===ownId(); }
  function store(key,value){ try{ localStorage.setItem(key,value); }catch(error){} }
  function recall(key){ try{ return localStorage.getItem(key)||''; }catch(error){ return ''; } }
  function message(text, error=false) { $('notice').textContent=text; $('notice').className='visible'+(error?' error':''); }
  function hideMessage() { $('notice').className=''; }
  async function api(path, options={}) {
    const response=await fetch(path, {cache:'no-store',...options});
    const data=await response.json();
    if (!response.ok || data.error) throw new Error(data.error || '请求失败');
    return data;
  }
  async function command(action, values={}, tries=250) {
    const job=await api('/api/command',{method:'POST',headers:{'Content-Type':'application/json','X-Sandbox-Token':P.catalog.token},
      body:JSON.stringify({action,...values})});
    for (let i=0;i<tries;i++) {
      await sleep(40);
      const receipt=await api('/api/command?id='+job.id);
      if(receipt.done) return receipt.result;
    }
    throw new Error('命令仍在处理');
  }
  async function act(fn) {
    if(P.busy) return;
    P.busy=true;
    render();
    try { hideMessage(); await fn(); await poll(); }
    catch(error) { message(error.message,true); }
    finally {P.busy=false; render();}
  }
  function members() { return P.scene.teams.flatMap((t,team)=>t.map((p,slot)=>({...p,team,slot}))); }
  function selectedDraft() { const row=members()[P.selected]; return row ? P.scene.teams[row.team][row.slot] : null; }
  function planes() {
    if(P.state.status!=='setup') return P.state.planes.filter(p=>P.showAI||!p.ai);
    if(P.mode==='replay') return [];
    return members().map((p,id)=>({id,team:p.team,aircraft:p.aircraft,alive:true,position_m:p.position_m,
      heading_deg:(Math.atan2(p.velocity_mps[0],p.velocity_mps[1])*180/Math.PI+360)%360,
      speed_mps:Math.hypot(...p.velocity_mps),missiles:p.missiles,missile:p.missile,chaff:p.chaff,
      phase:'准备',mode:'auto',order:null,tracks:[]}));
  }
  function selected() {return planes().find(p=>p.id===P.selected);}
  function option(select,value,label) { const o=document.createElement('option');o.value=value;o.textContent=label;select.append(o); }
  function fillMissiles(aircraft, selectedId) {
    const eq=P.catalog.aircraft.find(a=>a.id===aircraft);
    $('missile').replaceChildren(); option($('missile'),'','无挂载');
    for(const m of eq.missiles) option($('missile'),m.id,missileName(m.id)+' · 最多 '+m.max+' 枚');
    $('missile').value=selectedId || '';
    $('chaff').max=eq.chaff_max;
    const m=eq.missiles.find(m=>m.id===selectedId);
    $('ammo').max=m?m.max:0;
  }
  function setField(id,value) {if(document.activeElement!==$(id)) $(id).value=value;}
  function rebuildForm() {
    const p=selected();if(!p)return;
    const setup=P.state.status==='setup', draft=setup?selectedDraft():null;
    if(setup) {
      $('aircraft').value=draft.aircraft;
      fillMissiles(draft.aircraft,draft.missile);
      $('ammo').value=draft.missiles ?? 0;
      $('chaff').value=draft.chaff ?? ''; $('chaff').placeholder='随机';
      $('east').value=fmt(draft.position_m[0]/1000,2);$('north').value=fmt(draft.position_m[1]/1000,2);
      $('archetype').value=draft.archetype||'middle';$('skill').value=draft.skill||'top';
    }
    $('mode').value=p.mode;
    const order=p.order;
    $('altitude').value=fmt(order?order.altitude_m:p.position_m[2]);
    $('velocity').value=fmt(order?order.speed_mps:p.speed_mps);
    $('heading').value=fmt(order?order.heading_deg:p.heading_deg,1);
  }
  // Ctrl / Cmd + click adds or removes an aircraft (not while deploying); P.selected stays the primary (last chosen).
  function select(id,add=false) {
    releaseKeys();
    if(add&&P.state&&P.state.status!=='setup'){const i=P.sel.indexOf(id);if(i<0)P.sel.push(id);else if(P.sel.length>1)P.sel.splice(i,1);}
    else P.sel=[id];
    P.selected=P.sel.at(-1);P.formSignature='';render();
  }
  function selection() {
    const ps=planes();if(!P.sel.includes(P.selected))P.sel=[P.selected];
    if(P.state?.status==='setup')P.sel=[P.selected];
    P.sel=P.sel.filter(id=>ps.some(p=>p.id===id));
    return P.sel.map(id=>ps.find(p=>p.id===id));
  }
  // A <select> refreshed every poll keeps its options (and an open drop-down) while the set of values is unchanged.
  function syncOptions(sel,items,empty){
    const sig=items.map(i=>i.value).join('|')||'-';
    if(sel.dataset.sig!==sig){const prev=sel.value;sel.replaceChildren();if(!items.length)option(sel,'',empty);
      for(const i of items)option(sel,i.value,i.text);if(items.some(i=>String(i.value)===prev))sel.value=prev;sel.dataset.sig=sig;}
    else items.forEach((i,k)=>{if(sel.options[k]&&sel.options[k].textContent!==i.text)sel.options[k].textContent=i.text;});
  }
  const trackItems=p=>(p.tracks||[]).map(c=>({value:String(c.track),text:'航迹 '+c.track+' · '+fmt(c.range_m/1000,1)+' km · '+fmt(c.bearing_deg)+'°'}));
  function editDraft() {
    if(P.state.status!=='setup')return;
    const p=selectedDraft();if(!p)return;
    p.aircraft=$('aircraft').value;p.missile=$('missile').value||null;
    p.missiles=Number($('ammo').value);
    if($('chaff').value==='')delete p.chaff;else p.chaff=Number($('chaff').value);
    const h=Number($('heading').value)*Math.PI/180, speed=Number($('velocity').value);
    p.position_m=[Number($('east').value)*1000,Number($('north').value)*1000,Number($('altitude').value)];
    p.velocity_mps=[Math.sin(h)*speed,Math.cos(h)*speed,0];
    p.archetype=$('archetype').value;p.skill=$('skill').value;
    P.dirty=true;render();
  }
  async function configure() {
    if(P.state.status!=='setup')return;
    P.scene.seed=Number($('seed').value);P.scene.time_limit_s=Number($('limit').value);
    await command('configure',{scenario:P.scene});P.dirty=false;
  }
  // The lists are refreshed on every state poll (~20 Hz). Rows are kept and updated in place by key: a button
  // replaced between press and release would swallow the click.
  function rowsFor(list,keys){
    const old=new Map([...list.children].filter(r=>r.dataset.key!=null).map(r=>[r.dataset.key,r]));
    const rows=keys.map(k=>{let r=old.get(String(k));if(!r){r=document.createElement('button');r.dataset.key=String(k);
      const a=document.createElement('span'),b=document.createElement('span');a.className='row-title';b.className='row-meta';r.append(a,b);}return r;});
    if(rows.length!==list.children.length||rows.some((r,i)=>list.children[i]!==r))list.replaceChildren(...rows);
    return rows;
  }
  function render() {
    if(!P.state)return;
    const s=P.state, setup=s.status==='setup', done=s.status==='completed', running=s.status==='running';
    const playback=s.status==='playback', playing=!!s.replay?.playing, replayUI=P.mode==='replay';
    const status={setup:'布置中',running:'推演中',paused:'已暂停',stepping:'单步中',completed:'推演结束',error:'运行错误',playback:playing?'WT 回放 · 播放中':'WT 回放 · 已暂停'};
    $('status').textContent=(s.fork?'接管推演 · ':'')+status[s.status];$('setup-tag').textContent=setup?'布置中':'已锁定';
    $('clock').textContent=clockText(s.time_s+(s.fork?.offset||0));
    $('speed').value=s.speed;$('actual').textContent='实际 '+fmt(s.actual_speed,1)+'×';
    $('start').textContent=playback?(playing?'▶ 播放中':'▶ 播放'):setup?'▶ 开始':running?'▶ 推演中':'▶ 继续';
    $('start').disabled=P.busy||running||done||s.status==='error'||s.status==='stepping'||playback&&playing||setup&&replayUI;
    $('pause').disabled=P.busy||!(['running','stepping'].includes(s.status)||playback&&playing);
    $('step').disabled=P.busy||!['setup','paused'].includes(s.status)||setup&&replayUI;
    $('reset').disabled=P.busy||setup||playback;$('reset').textContent=s.fork?'↺ 返回回放':'↺ 重置';$('preset').disabled=!setup||P.busy;
    $('seed').disabled=$('limit').disabled=!setup||P.busy;
    $('save').disabled=$('speed').disabled=P.busy;$('save').disabled=P.busy||playback||setup&&replayUI;
    $('mode-setup').className=replayUI?'':'active';$('mode-replay').className=replayUI?'active':'';
    $('mode-setup').disabled=$('mode-replay').disabled=P.busy;
    $('replay-side').hidden=!replayUI||!!s.fork;$('setup-side').hidden=replayUI&&!s.fork;$('scene-controls').hidden=!!s.fork;
    renderReplay();
    const count=setup&&replayUI?[0,0]:setup?P.scene.teams.map(t=>t.length):s.teams_alive;
    $('score').replaceChildren();count.forEach((n,i)=>{const e=document.createElement('span');e.className=i?'orange':'blue';e.textContent=teams[i]+' '+n;$('score').append(e);});
    $('add-blue').disabled=!setup||P.busy||count[0]>=P.catalog.max_team;$('add-orange').disabled=!setup||P.busy||count[1]>=P.catalog.max_team;
    const ps=planes();
    for(const team of [0,1]) {
      const members=ps.filter(p=>p.team===team),rows=rowsFor($(team?'orange-list':'blue-list'),members.map(p=>p.id));
      members.forEach((p,i)=>{
        const row=rows[i];row.className='unit-row'+(team?' orange-row':'')+(p.id===P.selected?' selected':'')+(!p.alive?' dead':'');
        row.children[0].textContent=p.player?short(p.player)+' · '+title(p.aircraft):'#'+p.id+' '+title(p.aircraft);
        row.children[1].textContent=(p.missile?missileName(p.missile)+' · ':'')+(p.alive?fmt(p.position_m[2])+' m · '+fmt(p.speed_mps)+' m/s · '+(phases[p.phase]||p.phase||'接敌'):'已损失');
        row.onclick=e=>select(p.id,e.ctrlKey||e.metaKey);
      });
    }
    const chosen=selection(),multi=chosen.length>1,p=multi?null:selected();
    $('empty-selection').hidden=chosen.length>0;$('unit-panel').hidden=!p;$('group-panel').hidden=!multi;
    if(multi)renderGroup(chosen,{playback,done});
    if(p) {
      $('unit-id').textContent=teams[p.team]+' #'+p.id;$('unit-name').textContent=p.player?short(p.player)+' · '+title(p.aircraft):title(p.aircraft);$('unit-name').className='unit-name '+(p.team?'orange':'blue');
      $('unit-state').replaceChildren();
      for(const text of ['高度 '+fmt(p.position_m[2])+' m','速度 '+fmt(p.speed_mps)+' m/s','航向 '+fmt(p.heading_deg,1)+'°','导弹 '+fmt(p.missiles)+' 枚',p.chaff==null?(playback?'箔条 未知':'箔条 随机'):'箔条 '+fmt(p.chaff)+' 束',p.alive?'存活':'已损失']) {const t=document.createElement('span');t.textContent=text;$('unit-state').append(t);}
      const locked=playback||!!p.tracked;
      $('edit-panel').hidden=!setup;$('mode-panel').hidden=setup||locked;$('nav-fields').hidden=$('apply').hidden=locked;$('track-note').hidden=!locked;
      if(locked)$('track-note').textContent=playback?(p.controllable?'可接管 · '+(p.fork_missile?missileName(p.fork_missile):'无导弹'):'不可接管：'+(p.control_reason||'—')):'按 WT 轨迹'+(p.release_at!=null?'，'+fmt(p.release_at,2)+' s 起接管':'');
      $('remove').hidden=!setup;$('remove').disabled=P.busy||setup&&count[p.team]<=1;
      $('apply').textContent=setup?'应用':'下达';
      $('apply').disabled=P.busy||!p.alive||done||s.status==='error'||p.mode==='pilot';
      $('destination').hidden=$('pilot').hidden=setup||locked;
      $('destination').disabled=$('pilot').disabled=P.busy||!p.alive||done||s.status==='error';
      $('pilot').textContent=p.mode==='pilot'?'退出驾驶':'键盘驾驶';
      $('mode').disabled=P.busy||!p.alive||done||s.status==='error';
      $('mode').value=p.mode;
      $('mode-help').textContent=p.mode==='auto'?'脚本控制飞行和武器。':p.mode==='pilot'?'点击战场后用键盘驾驶，武器手动。':'给航向或目的地，武器手动。';
      const signature=p.id+'|'+setup+'|'+p.mode;
      if(P.formSignature!==signature){rebuildForm();P.formSignature=signature;}
      // The own aircraft's weapons live in the 本机 section.
      $('weapon-panel').hidden=setup||locked||p.mode==='auto'||isMe(p);
      syncOptions($('tracks'),trackItems(p),'无雷达航迹');
      $('fire').disabled=P.busy||done||!p.alive||!(p.tracks||[]).length||p.missiles<=0;
      $('drop-chaff').disabled=P.busy||done||!p.alive||p.chaff<=0;
      $('make-me').hidden=setup||isMe(p)||playback&&!p.player||!playback&&(!p.alive||done);
    }
    const q=selected();$('control-banner').hidden=!q||q.mode!=='pilot'||!q.alive||done;
    if(q&&q.mode==='pilot')$('control-banner').textContent=(document.activeElement===canvas?'驾驶中':'点击战场开始驾驶')+' · W/S 俯仰 · A/D 滚转 · R/F 油门 · 空格 减速板 · L 改平 · 拖动环视 · C 回正';
    renderOwn({setup,playback,done});
    $('deployment-help').textContent=P.pending?.type==='deploy'?'点击地图放置'+teams[P.pending.team]+'飞机，Esc 取消。':setup?'点选飞机编辑，拖动改位置。':'点选飞机下令或驾驶。';
    $('health').textContent=s.missile_errors||s.fm_faults?'导弹错误 '+s.missile_errors+' · 飞行模型越界 '+s.fm_faults:'';
    $('health').style.color=s.missile_errors||s.fm_faults?'#ff979a':'';
    $('terminal').hidden=!done;
    if(done){$('terminal').replaceChildren();const strong=document.createElement('strong');strong.textContent=s.output?.winner_team==null?'推演结束 · 平局':teams[s.output.winner_team]+'获胜';const sub=document.createElement('div');sub.textContent=(reasons[s.reason]||s.reason)+' · '+fmt(s.time_s,2)+' s';$('terminal').append(strong,sub);}
    const note=done?'场景、结果和回放已保存。':'';$('run-note').textContent=note;$('run-note').parentElement.hidden=!note;
    if(s.error)message(s.error,true);
    if(s.output){const o=$('outputs');o.replaceChildren();const t=document.createElement('p');t.textContent='记录：'+({completed:'已完成',partial:'尚未结束',cancelled:'已中止',server_stopped:'服务关闭',error:'运行错误'}[s.output.status]||s.output.status);o.append(t);for(const [file,url]of Object.entries(s.output.links)){const a=document.createElement('a');a.href=url;a.textContent=file==='result.json'?'结果 JSON':file==='scenario.json'?'场景 JSON':'回放 JSONL';o.append(a);}}
    renderEvents();
  }
  function renderReplay(){
    const s=P.state,list=$('replay-list');
    if(!P.replays.length){if(!list.querySelector('.help')){const e=document.createElement('p');e.className='help';e.textContent='没有回放（outputs/engagements/wt_real、outputs/sandbox）';list.replaceChildren(e);}}
    else{const rows=rowsFor(list,P.replays.map(r=>r.file));
      P.replays.forEach((r,i)=>{const row=rows[i];row.className='unit-row replay-row'+(s.replay_file===r.file?' selected':'');
        row.children[0].textContent=(r.kind==='wt'?'WT · ':'沙盘 · ')+(r.kind==='wt'?r.file.split('/').at(-1).replace(/\.jsonl$/,''):r.file.split('/').at(-2));
        row.children[1].textContent=new Date(r.mtime*1000).toLocaleString('zh-CN',{hour12:false})+' · 时长 '+(r.duration_s==null?'—':clockText(r.duration_s))+' · 玩家 '+r.players+(r.units>r.players?' · AI '+(r.units-r.players):'');
        row.disabled=P.busy||!!s.fork;row.onclick=()=>act(async()=>{await command('load_replay',{file:r.file});});});}
    renderWrpl();
    $('replay-tag').textContent=!s.replay_file?'未加载':s.fork?'接管推演':s.replay?.playing?'播放中':'已暂停';
    const ro=s.replay?.roster;
    if(ro&&P.meFile!==s.replay.file){P.meFile=s.replay.file;const sel=$('me');sel.replaceChildren();option(sel,'','（未选择）');
      const players=Object.values(ro).filter(r=>!r.ai).sort((a,b)=>a.team-b.team||a.player.localeCompare(b.player));
      for(const r of players)option(sel,r.player,short(r.player)+' · '+title(r.aircraft)+' · '+teams[r.team]);
      const saved=recall('wt-sandbox-me');P.me=players.some(r=>r.player===saved)?saved:'';sel.value=P.me;}
    $('me').disabled=!ro||P.busy;$('show-ai').checked=P.showAI;
    $('fork').disabled=P.busy||s.status!=='playback'||!!s.replay?.playing;
    $('fork').title=s.replay?.playing?'先暂停':'';
    const box=$('fork-info');box.hidden=!s.fork;
    if(s.fork){const f=s.fork;box.replaceChildren();const h=document.createElement('strong');h.textContent='WT 回放接管 · '+f.file;box.append(h);
      for(const text of ['接管时刻 '+clockText(f.t_fork),'接管：'+(f.released.length?f.released.map(r=>short(r.player)+'（'+(controlModes[r.mode]||r.mode)+'）').join('、'):'无'),...(f.skipped?['未重飞 '+f.skipped+' 枚（见事件）']:[]),...(f.notes||[])]){const e=document.createElement('p');e.textContent=text;box.append(e);}}
    $('timeline').hidden=!s.replay;
    if(s.replay){const r=s.replay,range=$('tl-range');range.min=r.start_s;range.max=r.duration_s;if(!P.seeking)range.value=s.time_s;range.disabled=P.busy;
      $('tl-label').textContent=clockText(P.seeking?Number(range.value):s.time_s)+' / '+clockText(r.duration_s);
      if(P.markerFile!==r.file){P.markerFile=r.file;const marks=$('tl-markers'),span=Math.max(1e-6,r.duration_s-r.start_s);marks.replaceChildren();
        for(const m of r.markers){const e=document.createElement('span');e.className='tl-mark '+m.kind;e.textContent=m.kind==='launch'?'▲':'✕';e.style.left=clamp((m.t-r.start_s)/span*100,0,100)+'%';e.title=clockText(m.t)+' · '+m.text;marks.append(e);}}}
  }
  // .wrpl import (docs/wrpl_import_spec.md §5): uploads queue on the server; progress comes with /api/state.
  const wrplPhases={saving:'上传中',queued:'排队',parse:'解析中',convert:'转换中',done:'完成',error:'失败',needs_game:'需要游戏目录',
    needs_backend:'需要解析后端',downloading:'下载中',cancelled:'已取消'};
  function renderWrpl(){
    const w=P.state?.wrpl;if(!w)return;
    const items=w.items.filter(i=>!i.done||P.wrplMine.has(i.id)).slice(-4);
    $('wrpl-box').hidden=!items.length;$('wrpl-pick').disabled=!!P.state.fork;
    const sig=JSON.stringify([items,P.busy]);
    if(sig!==P.wrplSig){P.wrplSig=sig;const rows=$('wrpl-rows');rows.replaceChildren();
      for(const i of items){const row=document.createElement('div');row.className='wrpl-row'+(i.phase==='error'?' error':'');
        const t=document.createElement('div');const pct=(i.phase==='saving'||i.phase==='downloading')&&i.total?' '+Math.min(100,Math.round(i.received/i.total*100))+'%':'';
        t.textContent=i.name+' · '+(wrplPhases[i.phase]||i.phase)+pct;row.append(t);
        if(i.message){const m=document.createElement('small');m.textContent=i.message;row.append(m);}
        if(i.log&&i.log.length){const pre=document.createElement('pre');pre.textContent=i.log.join('\n');row.append(pre);}
        const file=P.wrplMine.get(i.id),buttons=[];
        if(i.done&&i.duplicate&&file)buttons.push(['重新导入',()=>importFiles([file],true)]);
        if(i.phase==='error'&&file)buttons.push(['重试',()=>importFiles([file],true)]);
        if(!i.done&&i.phase!=='saving')buttons.push(['取消',()=>act(()=>command('wrpl_cancel',{id:i.id}))]);
        for(const [text,fn] of buttons){const b=document.createElement('button');b.textContent=text;b.disabled=P.busy;b.onclick=fn;row.append(b);}
        rows.append(row);}}
    const head=w.items.find(i=>!i.done&&i.phase!=='saving');
    $('wrpl-game').hidden=head?.phase!=='needs_game';$('wrpl-dl').hidden=head?.phase!=='needs_backend';
    $('wrpl-game-ok').disabled=$('wrpl-dl-go').disabled=P.busy;
    if(head?.phase==='needs_game'&&!$('wrpl-game-path').value&&w.game_dir&&document.activeElement!==$('wrpl-game-path'))$('wrpl-game-path').value=w.game_dir;
    for(const i of items)if(i.phase==='done'&&i.file&&!P.wrplLoaded.has(i.id)){P.wrplLoaded.add(i.id);afterImport(i);}
  }
  async function afterImport(i){
    await loadReplays();const s=P.state;
    if(P.mode!=='replay'||!['setup','playback'].includes(s.status)||P.busy||!P.replays.some(r=>r.file===i.file))return;
    try{await command('load_replay',{file:i.file});message((i.duplicate?'已导入过，已定位：':'已导入：')+i.file.split('/').at(-1).replace(/\.jsonl$/,''));}
    catch(error){message(error.message,true);}
  }
  async function uploadWrpl(file,force){
    const max=P.state?.wrpl?.max_bytes;if(max&&file.size>max)throw new Error('文件超过 '+Math.round(max/1048576)+' MB 上限');
    const response=await fetch('/api/wrpl?name='+encodeURIComponent(file.name)+(force?'&force=1':''),{method:'POST',cache:'no-store',
      headers:{'Content-Type':'application/octet-stream','X-Sandbox-Token':P.catalog.token},body:file});
    let data={};try{data=await response.json();}catch(error){}
    if(!response.ok||data.error)throw new Error(data.error||'上传失败');
    P.wrplMine.set(data.id,file);
  }
  async function importFiles(files,force=false){
    const list=[...files];
    if(!list.length)return;
    if(list.some(f=>!/\.wrpl$/i.test(f.name)))message('只接受 .wrpl 文件',true);
    for(const f of list.filter(f=>/\.wrpl$/i.test(f.name))){try{await uploadWrpl(f,force);}catch(error){message(f.name+'：'+error.message,true);}}
  }
  function openFork(){
    const s=P.state;if(s.status!=='playback'||s.replay?.playing)return;
    $('fork-title').textContent='从 '+clockText(s.time_s)+' 接管';const rows=$('fork-rows');rows.replaceChildren();
    for(const p of s.planes.filter(p=>p.alive&&!p.ai).sort((a,b)=>a.team-b.team||a.id-b.id)){
      const row=document.createElement('div');row.className='fork-row';const label=document.createElement('div');
      const n=document.createElement('span');n.className=p.team?'orange':'blue';n.textContent=short(p.player)+(isMe(p)?'（我）':'');
      const sub=document.createElement('small');sub.textContent=title(p.aircraft)+' · '+(p.controllable?(p.fork_missile?missileName(p.fork_missile)+' × '+fmt(p.missiles):'无导弹'):'不可接管：'+p.control_reason);label.append(n,sub);
      const sel=document.createElement('select');sel.dataset.id=p.id;for(const [k,v] of Object.entries(controlModes))option(sel,k,v);
      sel.value=p.controllable&&isMe(p)?'pilot':'track';sel.disabled=!p.controllable;row.append(label,sel);rows.append(row);}
    if(!rows.children.length){const e=document.createElement('p');e.className='help';e.textContent='此刻没有存活的玩家飞机';rows.append(e);}
    $('fork-ai').checked=P.showAI;$('fork-panel').hidden=false;
  }
  function eventText(e) {
    const plane=e.plane??e.shooter, who=plane==null?'':nameOf(plane)+' ';
    if(e.kind==='launch')return who+'发射'+(inReplay()&&e.missile?' '+missileName(baseMissile(e.missile)):'导弹')+' #'+e.uid+' → '+nameOf(e.target);
    if(e.kind==='missile_end'&&e.outcome)return '导弹 #'+e.uid+' · '+(outcomes[e.outcome]||e.outcome);
    if(e.kind==='kill'&&inReplay())return nameOf(e.killer)+' 击落 '+nameOf(e.victim);
    if(e.kind==='replay_shot_skipped')return '未重飞 · '+missileName(baseMissile(e.missile))+' · '+e.reason;
    if(e.kind==='replay_release')return who+'接管 · '+(controlModes[e.mode]||e.mode);
    if(e.kind==='rwr')return who+(e.warning==='missile'?'导弹告警':'雷达锁定告警');
    if(e.kind==='datalink_lost')return '导弹 #'+e.uid+' 数据链中断 · '+({track_lost:'载机丢失航迹',shooter_dead:'载机已损失',seeker_track:'导引头已截获'}[e.reason]||e.reason);
    if(e.kind==='seeker_on')return '导弹 #'+e.uid+' 导引头开启';
    if(e.kind==='missile_end')return '导弹 #'+e.uid+' 终止 · '+({lifetime:'飞行时间耗尽',fuse:'近炸',ground:'触地',error:'模型错误',target_dead:'目标已损失'}[e.result]||e.result);
    if(e.kind==='death')return nameOf(e.plane)+' 损失 · '+({missile:'导弹击落',crash:'坠毁',out_of_bounds:'出界',left_replay:'离开回放'}[e.cause]||(String(e.cause).startsWith('replay_')?'回放记录 · '+e.cause.slice(7):e.cause));
    if(e.kind==='chaff')return who+'投放箔条';
    if(e.kind==='damage')return (e.by!=null?nameOf(e.by)+' 对 ':'')+nameOf(e.plane)+' 造成'+(e.severity==='critical'?'致命':'严重')+'损伤'+(e.fire?'（起火）':'');
    if(e.kind==='phase'&&['evade','recommit','round2'].includes(e.to))return who+(phases[e.to]||e.to);
    if(e.kind==='human_order')return who+({auto:'切回自主脚本',manual:'切到人工导航',pilot:'进入驾驶'}[e.mode]);
    if(e.kind==='human_input')return who+'操纵输入 · 滚转 '+e.keys.roll+' / 俯仰 '+e.keys.pitch;
    if(e.kind==='waypoint_arrived')return who+'到达目的地';
    if(e.kind==='observed_contact'&&e.source==='radar'&&e.change==='appeared')return who+'雷达发现目标';
    if(e.kind==='end')return '推演结束 · '+(reasons[e.reason]||e.reason);
    if(e.kind==='missile_error')return '导弹模型错误'+(e.error?' · '+e.error:'');
    return null;
  }
  // Events go to three lists: 全局 (kills, losses, damage, end, system notes), the selected aircraft and 本机. Key-by-key
  // manual input is not listed. Lists only append (oldest first) and stay scrolled to the bottom unless scrolled up.
  const GLOBAL_KINDS=new Set(['kill','death','damage','end','missile_error','replay_shot_skipped','replay_release']);
  function concerns(e){
    const ids=new Set();for(const k of ['plane','shooter','killer','victim','by'])if(e[k]!=null)ids.add(e[k]);
    if(e.kind==='launch'&&e.target!=null)ids.add(e.target);
    if(e.uid!=null&&e.kind!=='launch'){const o=P.missileOwner.get(e.uid);if(o){ids.add(o.shooter);if(o.target!=null)ids.add(o.target);}}
    return ids;
  }
  function eventRows(entries){
    return entries.map(([k,e])=>({key:k,cls:e.kind==='death'||e.kind==='kill'||e.kind==='damage'?'hit':e.kind==='rwr'?'warning':e.kind.startsWith('human')?'order':'',
      time:fmt(e.t+(P.state.fork?.offset||0),1),text:e.text}));
  }
  function syncList(el,items,empty){
    if(!items.length){if(!(el.children.length===1&&el.firstChild.className==='help')){const h=document.createElement('p');h.className='help';h.textContent=empty;el.replaceChildren(h);}return;}
    const keys=new Set(items.map(i=>String(i.key))),bottom=el.scrollHeight-el.scrollTop-el.clientHeight<12;
    let kept=0;for(const c of [...el.children]){if(c.dataset.key!=null&&keys.has(c.dataset.key))kept++;else c.remove();}
    const have=new Set([...el.children].map(c=>c.dataset.key));
    for(const i of items){if(have.has(String(i.key)))continue;const row=document.createElement('div');row.dataset.key=String(i.key);row.className='event '+i.cls;
      const t=document.createElement('time');t.textContent=i.time;const x=document.createElement('span');x.textContent=i.text;row.append(t,x);el.append(row);}
    if(bottom||kept<items.length/2)el.scrollTop=el.scrollHeight;
  }
  function renderEvents(){
    const s=P.state,start=s.event_total-s.events.length;
    s.events.forEach((e,i)=>{if(e.kind==='launch'&&e.uid!=null)P.missileOwner.set(e.uid,{shooter:e.shooter??e.plane,target:e.target});
      const t=eventText(e);if(t)P.events.set(start+i,{...e,text:t});});
    if(P.events.size>400)for(const k of [...P.events.keys()].slice(0,P.events.size-400))P.events.delete(k);
    const all=[...P.events.entries()].filter(([,e])=>e.kind!=='human_input'),global=all.filter(([,e])=>GLOBAL_KINDS.has(e.kind));
    $('global-count').textContent=global.length+' 条';
    syncList($('global-events'),eventRows(global.slice(-120)),'暂无事件');
    const picked=new Set(selection().map(p=>p.id)),own=ownId();
    syncList($('unit-events'),eventRows(all.filter(([,e])=>[...concerns(e)].some(id=>picked.has(id))).slice(-40)),'暂无事件');
    syncList($('own-events'),own==null?[]:eventRows(all.filter(([,e])=>concerns(e).has(own)).slice(-80)),'暂无事件');
  }
  function renderGroup(list,{playback,done}){
    const rows=rowsFor($('group-rows'),list.map(p=>p.id)),cmd=list.filter(p=>p.alive&&!p.tracked&&!playback);
    list.forEach((p,i)=>{const row=rows[i];row.className='unit-row'+(p.team?' orange-row':'')+(p.id===P.selected?' selected':'')+(!p.alive?' dead':'');
      row.children[0].textContent=(p.player?short(p.player):'#'+p.id)+' · '+title(p.aircraft)+(isMe(p)?' · 本机':'');
      row.children[1].textContent=p.alive?fmt(p.position_m[2])+' m · '+fmt(p.speed_mps)+' m/s · '+(p.tracked?'WT 轨迹':controlModes[p.mode]||p.mode||'—'):'已损失';
      row.onclick=e=>select(p.id,e.ctrlKey||e.metaKey);});
    $('unit-id').textContent='已选 '+list.length+' 架';
    $('group-note').textContent=playback?'回放中只能查看':cmd.length===list.length?'':cmd.length+' 架可下令';
    const sig=list.map(p=>p.id).join(',');
    if(P.groupSig!==sig){const lead=selected()||list[0];$('g-altitude').value=Math.round(lead.position_m[2]/100)*100;$('g-velocity').value=Math.round(lead.speed_mps/10)*10;$('g-heading').value=Math.round(lead.heading_deg||0);$('g-mode').value='';P.groupSig=sig;}
    for(const id of ['g-mode','g-altitude','g-velocity','g-heading','g-apply','g-destination'])$(id).disabled=P.busy||done||!cmd.length;
  }
  function renderOwn({setup,playback,done}){
    const id=ownId(),p=id==null?null:(P.state.planes||[]).find(q=>q.id===id);
    $('own-empty').hidden=!!p;$('own-panel').hidden=!p;
    $('own-tag').textContent=p?(p.player?short(p.player)+' · ':'#'+p.id+' ')+title(p.aircraft):'未指定';
    if(!p){$('own-empty').textContent=setup?'开始后选中飞机点「设为本机」':playback?'在左侧「我的飞机」选择':'选中飞机点「设为本机」';return;}
    const st=$('own-state');st.replaceChildren();
    for(const text of ['高度 '+fmt(p.position_m[2])+' m','速度 '+fmt(p.speed_mps)+' m/s','航向 '+fmt(p.heading_deg,1)+'°','导弹 '+fmt(p.missiles)+' 枚',p.chaff==null?'箔条 未知':'箔条 '+fmt(p.chaff)+' 束',p.alive?(p.tracked||playback?'WT 轨迹':controlModes[p.mode]||p.mode):'已损失']){const t=document.createElement('span');t.textContent=text;st.append(t);}
    const inc=(P.state.missiles||[]).filter(m=>m.target===p.id).map(m=>Math.hypot(...m.position_m.map((v,i)=>v-p.position_m[i]))).sort((a,b)=>a-b);
    $('own-threat').hidden=!inc.length||!p.alive;if(inc.length)$('own-threat').textContent='来袭导弹 '+inc.length+' 枚 · 最近 '+fmt(inc[0]/1000,1)+' km';
    const can=!setup&&!playback&&!done&&p.alive&&!p.tracked,armed=can&&p.mode!=='auto';
    $('own-weapons').hidden=setup||playback||!!p.tracked;
    $('own-note').hidden=!can||armed;$('own-note').textContent='脚本控制武器，切到人工后可手动发射';
    syncOptions($('own-tracks'),trackItems(p),'无雷达航迹');
    $('own-tracks').disabled=!armed;
    $('own-fire').disabled=P.busy||!armed||!(p.tracks||[]).length||p.missiles<=0;$('own-chaff').disabled=P.busy||!armed||p.chaff<=0;
    $('own-pilot').hidden=!can;$('own-pilot').disabled=P.busy;$('own-pilot').textContent=p.mode==='pilot'?'退出驾驶':'驾驶本机';
  }

  async function poll(){
    const s=await api('/api/state');
    if(!P.scene||(!P.dirty&&s.status==='setup'))P.scene=clone(s.scenario);
    if(s.status==='playback'||s.fork)P.mode='replay';
    const fork=!!s.fork;if(fork!==P.lastFork||(s.replay_file||null)!==(P.state?.replay_file||null)){P.trails.clear();P.events.clear();P.missileOwner.clear();P.lastTime=-1;P.lastFork=fork;P.ownId=null;P.sel=[];}
    if(s.time_s<P.lastTime||s.status==='setup'&&P.lastStatus!=='setup'){P.trails.clear();P.events.clear();P.missileOwner.clear();P.lastTime=-1;P.formSignature='';}
    if(s.status==='setup'&&P.lastStatus!=='setup')P.ownId=null;
    if(s.time_s>P.lastTime){
      for(const p of s.planes.filter(p=>p.alive&&(P.showAI||!p.ai)))trail('p'+p.id,p.position_m,s.time_s,p.team);
      for(const m of s.missiles)trail('m'+m.uid,m.position_m,s.time_s,m.team);
      P.lastTime=s.time_s;
    }
    if(s.status!=='running'&&P.lastStatus==='running')releaseKeys();
    P.lastStatus=s.status;P.state=s;P.received=performance.now();terrainSync();render();
  }
  function trail(id,position,t,team){const tr=P.trails.get(id)||{points:[],team};tr.points.push({p:position.slice(),t});if(tr.points.length>240)tr.points.shift();P.trails.set(id,tr);}

  // True ENU perspective. All three axes use metres; the grid is z=0. Geometry is clipped
  // against a near plane in camera space so nothing behind the eye smears across the view.
  const NEAR=60;
  function cameraBasis(){
    const c=P.camera, horizontal=c.radius*Math.cos(c.elevation),eye=[c.x+horizontal*Math.sin(c.yaw),c.y-horizontal*Math.cos(c.yaw),c.z+c.radius*Math.sin(c.elevation)];
    const forward=unit([c.x-eye[0],c.y-eye[1],c.z-eye[2]]), right=unit(cross(forward,[0,0,1])),up=cross(right,forward);
    return{eye,forward,right,up,focal:Math.min(canvas.clientWidth,canvas.clientHeight)*1.08};
  }
  let B=null;
  function toCam(p){if(P.view==='2d')return p;const d=p.map((v,i)=>v-B.eye[i]);return[dot(d,B.right),dot(d,B.up),dot(d,B.forward)];}
  function fromCam(c){
    const w=canvas.clientWidth,h=canvas.clientHeight;
    if(P.view==='2d'){const scale=Math.min(w,h)/P.map.span;return{x:w/2+(c[0]-P.map.x)*scale,y:h/2-(c[1]-P.map.y)*scale,depth:1,scale};}
    const scale=B.focal/c[2];return{x:w/2+c[0]*scale,y:h/2-c[1]*scale,depth:c[2],scale};
  }
  function project(p){if(P.view==='3d'&&!B)B=cameraBasis();const c=toCam(p);if(P.view==='3d'&&c[2]<NEAR)return null;return fromCam(c);}
  function clipSeg(a,b){
    let ca=toCam(a),cb=toCam(b);
    if(P.view==='3d'){
      if(ca[2]<NEAR&&cb[2]<NEAR)return null;
      if(ca[2]<NEAR){const t=(NEAR-ca[2])/(cb[2]-ca[2]);ca=ca.map((v,i)=>v+(cb[i]-v)*t);}
      else if(cb[2]<NEAR){const t=(NEAR-cb[2])/(ca[2]-cb[2]);cb=cb.map((v,i)=>v+(ca[i]-v)*t);}
    }
    return[fromCam(ca),fromCam(cb)];
  }
  function clipPoly(points){
    let cs=points.map(toCam);
    if(P.view==='3d'){
      const out=[];
      for(let i=0;i<cs.length;i++){const a=cs[i],b=cs[(i+1)%cs.length],ia=a[2]>=NEAR,ib=b[2]>=NEAR;
        if(ia)out.push(a);
        if(ia!==ib){const t=(NEAR-a[2])/(b[2]-a[2]);out.push(a.map((v,k)=>v+(b[k]-v)*t));}}
      cs=out;
    }
    return cs.length>2?cs.map(fromCam):null;
  }
  function fog(depth){if(P.view==='2d')return 1;const far=Math.max(P.camera.radius*3,90000);return clamp(1.15-depth/far,.12,1);}
  function line(a,b,color,width=1,dash=[],alpha=1){
    const s=clipSeg(a,b);if(!s)return;
    ctx.globalAlpha=alpha*fog((s[0].depth+s[1].depth)/2);ctx.strokeStyle=color;ctx.lineWidth=width;ctx.setLineDash(dash);
    ctx.beginPath();ctx.moveTo(s[0].x,s[0].y);ctx.lineTo(s[1].x,s[1].y);ctx.stroke();ctx.setLineDash([]);ctx.globalAlpha=1;
  }
  function fadedLine(a,b,color,width,pieces=8,alpha=1){for(let k=0;k<pieces;k++){const f0=k/pieces,f1=(k+1)/pieces;line(a.map((v,i)=>v+(b[i]-v)*f0),a.map((v,i)=>v+(b[i]-v)*f1),color,width,[],alpha);}}
  function worldAt(x,y,altitude=0){
    const w=canvas.clientWidth,h=canvas.clientHeight;
    if(P.view==='2d'){const scale=Math.min(w,h)/P.map.span;return[P.map.x+(x-w/2)/scale,P.map.y-(y-h/2)/scale,altitude];}
    const b=cameraBasis(),ray=b.forward.map((v,i)=>v+(x-w/2)/b.focal*b.right[i]-(y-h/2)/b.focal*b.up[i]);
    if(Math.abs(ray[2])<1e-6)return null;
    const distance=(altitude-b.eye[2])/ray[2];if(distance<0)return null;
    return b.eye.map((v,i)=>v+ray[i]*distance);
  }
  // Snapshots arrive ~15 Hz; between them each body is advanced along its own velocity
  // (display only, capped at 0.25 s) so motion and the chase camera stay smooth.
  function lead(){const s=P.state;if(!s||!(s.status==='running'||s.replay?.playing))return 0;return clamp((performance.now()-P.received)/1000*s.speed,0,.25*s.speed);}
  function disp(o,dt){return o.velocity_mps&&o.alive!==false?o.position_m.map((v,i)=>v+o.velocity_mps[i]*dt):o.position_m;}

  // Terrain from the game's height rasters (/api/terrain). Contours arrive in Dagor X/Z; the sandbox frame is
  // x = X - ox, y = Z - oz with the replay's origin, or the map centre when there is no replay. Display only:
  // the flight and missile models do not see it.
  const T={choice:recall('wt-sandbox-terrain')||'auto',maps:[],want:null,key:'',geo:null,loading:null,error:null};
  function terrainWanted(){
    const auto=P.state?.terrain?.map||null,stem=T.choice==='auto'?auto:T.choice==='none'?null:T.choice;
    return stem&&T.maps.some(m=>m.map===stem)?stem:null;
  }
  function terrainSync(){
    const stem=terrainWanted(),origin=P.state?.terrain?.origin_dagor_xz||null,key=stem?stem+'|'+(origin?origin.join(','):'centre'):'';
    if(key===T.key)return;
    T.key=key;T.geo=null;T.error=null;
    if(!stem){legendTerrain();return;}
    T.loading=key;legendTerrain();
    api('/api/terrain?map='+encodeURIComponent(stem)).then(data=>{
      if(T.key!==key)return;
      T.geo=terrainGeometry(data,origin||data.center_dagor_xz);T.loading=null;legendTerrain();
    }).catch(error=>{if(T.key!==key)return;T.loading=null;T.error=error.message;legendTerrain();});
  }
  function terrainGeometry(d,origin){
    const[ox,oz]=origin,groups={under:[],minor:[],major:[],water:[]},paths={};
    for(const lv of d.levels)for(const line of lv.lines){const n=line.length/2,a=new Float32Array(n*3);
      for(let k=0;k<n;k++){a[3*k]=line[2*k]-ox;a[3*k+1]=line[2*k+1]-oz;a[3*k+2]=lv.h;}groups[lv.kind].push(a);}
    for(const kind in groups){const p=new Path2D();for(const a of groups[kind]){p.moveTo(a[0],a[1]);for(let k=3;k<a.length;k+=3)p.lineTo(a[k],a[k+1]);}paths[kind]=p;}
    const raw=atob(d.heights_i16le),nx=d.nx,nz=d.nz,hts=new Float32Array(nx*nz);
    for(let i=0;i<nx*nz;i++){const v=(raw.charCodeAt(2*i)|raw.charCodeAt(2*i+1)<<8)<<16>>16;hts[i]=v===-32768?NaN:v;}
    return{d,groups,paths,hts,nx,nz,cell:d.cell_m,x0:d.x0_m-ox,y0:d.z0_m-oz,image:hillshade(d,hts)};
  }
  // Muted hypsometric tint with light from the north-west; row 0 is the southern edge (the 2-D transform flips y).
  function hillshade(d,hts){
    const nx=d.nx,nz=d.nz,img=document.createElement('canvas');img.width=nx;img.height=nz;
    const g=img.getContext('2d'),data=g.createImageData(nx,nz),px=data.data,lo=d.height_min_m,span=Math.max(d.height_max_m-lo,1),water=d.water_m;
    const ramp=[[0,[22,46,42]],[.25,[38,70,52]],[.55,[78,86,60]],[.8,[104,96,76]],[1,[150,146,132]]];
    const tint=t=>{for(let k=1;k<ramp.length;k++)if(t<=ramp[k][0]){const[a,ca]=ramp[k-1],[b,cb]=ramp[k],f=(t-a)/(b-a);return ca.map((c,i)=>c+(cb[i]-c)*f);}return ramp.at(-1)[1];};
    const L=unit([-1,1,1.2]),ex=2.5/d.cell_m,at=(i,j)=>hts[clamp(i,0,nz-1)*nx+clamp(j,0,nx-1)];
    for(let i=0;i<nz;i++)for(let j=0;j<nx;j++){
      const h=hts[i*nx+j],o=4*(i*nx+j);if(h!==h){px[o+3]=0;continue;}
      let sx=(at(i,j+1)-at(i,j-1))*ex/2,sy=(at(i+1,j)-at(i-1,j))*ex/2;if(sx!==sx)sx=0;if(sy!==sy)sy=0;
      const shade=clamp((-sx*L[0]-sy*L[1]+L[2])/Math.hypot(sx,sy,1),0,1)*.75+.35;
      const c=water!=null&&h<water?[12,30,48]:tint((h-lo)/span);
      px[o]=c[0]*shade;px[o+1]=c[1]*shade;px[o+2]=c[2]*shade;px[o+3]=255;
    }
    g.putImageData(data,0,0);return img;
  }
  function groundAt(x,y){
    const t=T.geo;if(!t)return 0;const c=(x-t.x0)/t.cell,r=(y-t.y0)/t.cell,j=Math.floor(c),i=Math.floor(r);
    if(j<0||i<0||j>=t.nx-1||i>=t.nz-1)return 0;const f=c-j,e=r-i,n=t.nx,h=t.hts;
    const v=(h[i*n+j]*(1-f)+h[i*n+j+1]*f)*(1-e)+(h[(i+1)*n+j]*(1-f)+h[(i+1)*n+j+1]*f)*e;return v===v?v:0;
  }
  // "under": contours below the configured water level, usually sea floor, kept faint.
  const TERRAIN_STYLE=[['under','#5f8eb0',1,.14],['minor','#86b79c',1,.32],['major','#c4dcb6',1.35,.62],['water','#72bdf0',1.6,.8]];
  function drawTerrain(w,h){
    const t=T.geo;if(!t)return;const dpr=window.devicePixelRatio||1;
    if(P.view==='2d'){
      const s=Math.min(w,h)/P.map.span;ctx.save();ctx.setTransform(dpr*s,0,0,-dpr*s,dpr*(w/2-P.map.x*s),dpr*(h/2+P.map.y*s));
      ctx.globalAlpha=.85;ctx.imageSmoothingEnabled=true;ctx.drawImage(t.image,t.x0-t.cell/2,t.y0-t.cell/2,t.nx*t.cell,t.nz*t.cell);
      for(const[kind,color,width,alpha]of TERRAIN_STYLE){ctx.globalAlpha=alpha;ctx.strokeStyle=color;ctx.lineWidth=width/s;ctx.stroke(t.paths[kind]);}
      ctx.restore();return;
    }
    // 3-D: every contour at its own height, faded with distance in a few alpha bands (one stroke per band).
    const e=B.eye,r=B.right,u=B.up,f=B.forward,F=B.focal,cx=w/2,cy=h/2,far=Math.max(P.camera.radius*3,90000),bands=5;
    for(const[kind,color,width,alpha]of TERRAIN_STYLE){
      const paths=Array.from({length:bands},()=>new Path2D());
      for(const a of t.groups[kind]){let prev=false,lx=0,ly=0;
        for(let k=0;k<a.length;k+=3){const dx=a[k]-e[0],dy=a[k+1]-e[1],dz=a[k+2]-e[2],zc=dx*f[0]+dy*f[1]+dz*f[2];
          if(zc<NEAR){prev=false;continue;}
          const sx=cx+(dx*r[0]+dy*r[1]+dz*r[2])*F/zc,sy=cy-(dx*u[0]+dy*u[1]+dz*u[2])*F/zc;
          if(prev){const b=Math.min(bands-1,Math.floor((1-clamp(1.15-zc/far,.12,1))/.88*bands));paths[b].moveTo(lx,ly);paths[b].lineTo(sx,sy);}
          lx=sx;ly=sy;prev=true;}}
      ctx.strokeStyle=color;ctx.lineWidth=width;
      for(let b=0;b<bands;b++){ctx.globalAlpha=alpha*(1-(b+.5)/bands*.88);ctx.stroke(paths[b]);}
    }
    ctx.globalAlpha=1;
  }
  function legendTerrain(){
    const el=$('terrain-legend'),want=T.choice==='auto'?P.state?.terrain?.map:T.choice==='none'?null:T.choice;let text='';
    if(T.geo){const d=T.geo.d;text='地形 '+d.map+' · 等高线 '+d.interval_m+' m（粗线 '+d.major_m+' m）· '+fmt(d.height_min_m)+'–'+fmt(d.height_max_m)+' m'+(d.water_m!=null?' · 蓝线 水位 '+fmt(d.water_m)+' m':'');}
    else if(T.loading)text='地形加载中…';
    else if(T.error)text='地形读取失败：'+T.error;
    else if(want&&!T.maps.some(m=>m.map===want))text='地图 '+want+' 没有高度数据';
    el.textContent=text;el.hidden=!text;
  }
  function fillTerrainPicker(){
    const sel=$('terrain');sel.replaceChildren();option(sel,'auto','自动（回放地图）');option(sel,'none','不显示');
    for(const m of T.maps)option(sel,m.map,m.map+' · '+fmt(m.width_km)+' km');
    if(![...sel.options].some(o=>o.value===T.choice))T.choice='auto';sel.value=T.choice;
  }
  function bodyAxes(p){
    const H=p.heading_deg*Math.PI/180,Pt=(p.pitch_deg||0)*Math.PI/180,R=(p.roll_deg||0)*Math.PI/180;
    const f=[Math.sin(H)*Math.cos(Pt),Math.cos(H)*Math.cos(Pt),Math.sin(Pt)],r0=[Math.cos(H),-Math.sin(H),0],u0=cross(r0,f);
    const r=r0.map((v,i)=>v*Math.cos(R)-u0[i]*Math.sin(R));return{f,r,u:cross(r,f)};
  }
  // Symbol size relative to the original icons (aircraft and missiles); the chase view keeps its own size.
  const ICON=.2;
  const HULL=[[1.35,0],[.55,.1],[.05,.22],[-.15,.95],[-.42,.95],[-.38,.2],[-.72,.22],[-.78,.5],[-.98,.5],[-.92,0]];
  function drawAircraft(p,pos,chosen){
    const xy=project(pos);if(!xy)return null;
    const color=colors[p.team],ax=bodyAxes(P.view==='2d'?{heading_deg:p.heading_deg}:p),px=chosen&&p.mode==='pilot'&&$('follow').checked&&P.view==='3d'?44:(p.ai?11:17)*ICON,size=clamp(px/xy.scale,7,6000);
    const at=(x,y,z=0)=>pos.map((v,i)=>v+(ax.f[i]*x+ax.r[i]*y+ax.u[i]*z)*size);
    const outline=[...HULL,...HULL.slice(1,-1).reverse().map(([x,y])=>[x,-y])].map(([x,y])=>at(x,y));
    const poly=clipPoly(outline),fin=clipPoly([at(-.45,0),at(-.9,0,.42),at(-.98,0,.42),at(-.92,0)]);
    const toEye=P.view==='2d'?[0,0,1]:unit(B.eye.map((v,i)=>v-pos[i])),top=dot(ax.u,toEye)>=0;
    const me=isMe(p);ctx.globalAlpha=(p.alive?1:.3)*(p.ai?.55:1);
    if(chosen){ctx.shadowColor=color;ctx.shadowBlur=8;}
    if(poly){ctx.fillStyle=top?color:shade(color,.55);ctx.strokeStyle=chosen||me?'#ffffff':shade(color,1.35);ctx.lineWidth=(me?2.4:chosen?1.6:1)*Math.max(ICON,.5);ctx.lineJoin='round';
      ctx.beginPath();poly.forEach((q,i)=>i?ctx.lineTo(q.x,q.y):ctx.moveTo(q.x,q.y));ctx.closePath();ctx.fill();ctx.stroke();}
    ctx.shadowBlur=0;
    if(fin&&P.view==='3d'){ctx.fillStyle=shade(color,.75);ctx.beginPath();fin.forEach((q,i)=>i?ctx.lineTo(q.x,q.y):ctx.moveTo(q.x,q.y));ctx.closePath();ctx.fill();}
    ctx.globalAlpha=1;return xy;
  }
  function shade(hex,k){const n=parseInt(hex.slice(1),16),c=[n>>16,n>>8&255,n&255].map(v=>clamp(Math.round(v*k),0,255));return'rgb('+c.join(',')+')';}
  function chip(x,y,lines,color,strong){
    ctx.font=(strong?'600 ':'500 ')+'11px system-ui';const w1=ctx.measureText(lines[0]).width;ctx.font='10px ui-monospace,monospace';const w2=lines[1]?ctx.measureText(lines[1]).width:0;
    const w=Math.max(w1,w2)+14,h=lines[1]?32:19;P.labels.push([x,y-h/2,w,h]);
    ctx.fillStyle=strong?'rgba(18,32,46,.92)':'rgba(9,16,25,.72)';ctx.beginPath();ctx.roundRect(x,y-h/2,w,h,4);ctx.fill();
    ctx.fillStyle=color;ctx.fillRect(x,y-h/2+3,2,h-6);
    ctx.fillStyle=strong?'#ffffff':color;ctx.font=(strong?'600 ':'500 ')+'11px system-ui';ctx.fillText(lines[0],x+8,y+(lines[1]?-3:4));
    if(lines[1]){ctx.fillStyle='#9fb4c6';ctx.font='10px ui-monospace,monospace';ctx.fillText(lines[1],x+8,y+11);}
  }
  function backdrop(w,h){
    if(P.view==='2d'){const g=ctx.createRadialGradient(w/2,h/2,0,w/2,h/2,Math.max(w,h)*.75);g.addColorStop(0,'#13222f');g.addColorStop(1,'#091119');ctx.fillStyle=g;ctx.fillRect(0,0,w,h);return;}
    const dir=unit([B.forward[0],B.forward[1],0]),hy=clamp(h/2-dot(dir,B.up)/dot(dir,B.forward)*B.focal,-50,h+50);
    const sky=ctx.createLinearGradient(0,0,0,Math.max(hy,1));sky.addColorStop(0,'#08121d');sky.addColorStop(.7,'#13283a');sky.addColorStop(1,'#2a4a5e');
    ctx.fillStyle=sky;ctx.fillRect(0,0,w,Math.max(0,hy));
    const ground=ctx.createLinearGradient(0,hy,0,h);ground.addColorStop(0,'#1b2c33');ground.addColorStop(.25,'#111d22');ground.addColorStop(1,'#0a1216');
    ctx.fillStyle=ground;ctx.fillRect(0,Math.max(0,hy),w,h-Math.max(0,hy));
    const glow=ctx.createLinearGradient(0,hy-18,0,hy+10);glow.addColorStop(0,'rgba(120,180,200,0)');glow.addColorStop(.6,'rgba(120,180,200,.16)');glow.addColorStop(1,'rgba(120,180,200,0)');
    ctx.fillStyle=glow;ctx.fillRect(0,hy-18,w,28);
  }
  function hud(p,w,h){
    // Pilot instruments: attitude ball, speed / altitude tapes, load, throttle, threat list.
    const cx=w/2,cy=h-150,r=48,R=(p.roll_deg||0)*Math.PI/180,pitch=p.pitch_deg||0;
    ctx.save();ctx.beginPath();ctx.arc(cx,cy,r,0,Math.PI*2);ctx.clip();
    ctx.translate(cx,cy);ctx.rotate(-R);const off=pitch*1.6;
    ctx.fillStyle='#2d6c95';ctx.fillRect(-r*2,-r*2+off,r*4,r*2);ctx.fillStyle='#6b4a2c';ctx.fillRect(-r*2,off,r*4,r*2);
    ctx.strokeStyle='#ffffffcc';ctx.lineWidth=1;ctx.font='9px ui-monospace,monospace';ctx.fillStyle='#ffffffcc';
    for(let d=-30;d<=30;d+=10){if(!d)continue;const y=off-d*1.6;ctx.beginPath();ctx.moveTo(-14,y);ctx.lineTo(14,y);ctx.stroke();ctx.fillText(String(d),17,y+3);}
    ctx.strokeStyle='#fff';ctx.lineWidth=1.5;ctx.beginPath();ctx.moveTo(-r*2,off);ctx.lineTo(r*2,off);ctx.stroke();ctx.restore();
    ctx.strokeStyle='#ffd166';ctx.lineWidth=2.5;ctx.beginPath();ctx.moveTo(cx-30,cy);ctx.lineTo(cx-10,cy);ctx.lineTo(cx-5,cy+6);ctx.moveTo(cx+30,cy);ctx.lineTo(cx+10,cy);ctx.lineTo(cx+5,cy+6);ctx.stroke();
    ctx.strokeStyle='#9fc3d8';ctx.lineWidth=1.5;ctx.beginPath();ctx.arc(cx,cy,r,0,Math.PI*2);ctx.stroke();
    const box=(x,label,value,unit)=>{ctx.fillStyle='rgba(8,15,23,.82)';ctx.beginPath();ctx.roundRect(x,cy-26,96,52,6);ctx.fill();ctx.strokeStyle='#3d5d74';ctx.lineWidth=1;ctx.stroke();
      ctx.fillStyle='#8fa8bb';ctx.font='10px system-ui';ctx.fillText(label,x+9,cy-10);ctx.fillStyle='#eafaff';ctx.font='600 19px ui-monospace,monospace';ctx.fillText(value,x+9,cy+14);ctx.fillStyle='#8fa8bb';ctx.font='10px system-ui';ctx.fillText(unit,x+96-9-ctx.measureText(unit).width,cy+14);};
    box(cx-r-112,'速度',fmt(p.speed_mps),'m/s');box(cx+r+16,'高度',fmt(p.position_m[2]),'m');
    ctx.fillStyle='rgba(8,15,23,.82)';ctx.beginPath();ctx.roundRect(cx-34,cy-r-30,68,22,5);ctx.fill();ctx.fillStyle='#eafaff';ctx.font='600 13px ui-monospace,monospace';ctx.textAlign='center';ctx.fillText(fmt(p.heading_deg).padStart(3,'0')+'°',cx,cy-r-14);
    const g=p.load_g??1,thr=p.engine_percent??0,stats=['G '+fmt(g,1),'油门 '+fmt(thr)+'%','攻角 '+fmt(p.aoa_deg,1)+'°'];if(p.airbrake>.05)stats.push('减速板');
    ctx.font='11px ui-monospace,monospace';ctx.fillStyle=g>7?'#ff8a80':'#c9dcea';ctx.fillText(stats.join('   '),cx,cy+r+18);ctx.textAlign='left';
    const threats=P.state.missiles.filter(m=>m.target===p.id).map(m=>({m,d:Math.hypot(...m.position_m.map((v,i)=>v-p.position_m[i]))})).sort((a,b)=>a.d-b.d);
    if(threats.length){const t=threats[0],closing=t.m.velocity_mps?-dot(t.m.velocity_mps.map((v,i)=>v-p.velocity_mps[i]),unit(t.m.position_m.map((v,i)=>v-p.position_m[i]))):null;
      const text='导弹来袭 ×'+threats.length+' · 最近 '+fmt(t.d/1000,1)+' km'+(closing>0?' · 约 '+fmt(t.d/closing,0)+' s':'')+(t.m.seeker?' · 导引头已开':'');
      ctx.font='600 13px system-ui';const tw=ctx.measureText(text).width+24;const blink=Math.floor(performance.now()/350)%2;
      ctx.fillStyle=blink?'rgba(120,20,28,.92)':'rgba(80,14,20,.92)';ctx.beginPath();ctx.roundRect(cx-tw/2,14,tw,28,6);ctx.fill();ctx.fillStyle='#ffd7d4';ctx.textAlign='center';ctx.fillText(text,cx,33);ctx.textAlign='left';}
  }
  function draw(){
    const w=canvas.clientWidth,h=canvas.clientHeight,dpr=window.devicePixelRatio||1;
    if(canvas.width!==Math.round(w*dpr)||canvas.height!==Math.round(h*dpr)){canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr);}
    ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,w,h);
    if(!P.state){requestAnimationFrame(draw);return;}
    const half=mapHalf(),ps=planes(),sel=selected(),dt=lead(),picked=new Set(P.sel.length?P.sel:[P.selected]);
    if($('follow').checked&&sel){
      const at=disp(sel,dt);P.camera.x=at[0];P.camera.y=at[1];P.camera.z=at[2];P.map.x=at[0];P.map.y=at[1];
      if(sel.mode==='pilot'&&P.view==='3d'){const target=-sel.heading_deg*Math.PI/180+P.camera.orbit;let d=target-P.camera.yaw;d=Math.atan2(Math.sin(d),Math.cos(d));P.camera.yaw+=d*.12;}
    }
    B=P.view==='3d'?cameraBasis():null;
    backdrop(w,h);drawTerrain(w,h);
    // With terrain under it, the 3-D view drops the flat z=0 field and grid (they would show through the hills).
    const flat=!(T.geo&&P.view==='3d');
    const field=flat&&clipPoly([[-half,-half,0],[half,-half,0],[half,half,0],[-half,half,0]]);
    if(field){ctx.fillStyle=P.view==='3d'?'rgba(70,120,140,.07)':'rgba(70,120,140,.06)';ctx.beginPath();field.forEach((q,i)=>i?ctx.lineTo(q.x,q.y):ctx.moveTo(q.x,q.y));ctx.closePath();ctx.fill();}
    const spacing=P.view==='2d'&&P.map.span<40000?5000:10000;
    if(flat)for(let i=Math.ceil(-half/spacing)*spacing;i<=half;i+=spacing){const c=i===0?'#5d8aa3':'#36586b';fadedLine([i,-half,0],[i,half,0],c,1,8,.75);fadedLine([-half,i,0],[half,i,0],c,1,8,.75);}
    for(const[a,b]of [[[ -half,-half,0],[half,-half,0]],[[half,-half,0],[half,half,0]],[[half,half,0],[-half,half,0]],[[-half,half,0],[-half,-half,0]]])fadedLine(a,b,'#6f9bb3',1.6,8);
    for(const [axis,p]of [['N',[0,half*1.06,0]],['S',[0,-half*1.06,0]],['E',[half*1.06,0,0]],['W',[-half*1.06,0,0]]]){const xy=project(p);if(xy){ctx.globalAlpha=fog(xy.depth);ctx.fillStyle='#8fbad1';ctx.font='600 12px system-ui';ctx.textAlign='center';ctx.fillText(axis,xy.x,xy.y+4);ctx.textAlign='left';ctx.globalAlpha=1;}}
    if(sel&&sel.alive){const c=disp(sel,dt);for(const r of [10000,20000,40000]){const pts=[];for(let k=0;k<=72;k++){const a=k/72*Math.PI*2;const x=c[0]+Math.sin(a)*r,y=c[1]+Math.cos(a)*r;pts.push([x,y,groundAt(x,y)]);}
      for(let k=1;k<pts.length;k++)line(pts[k-1],pts[k],'#9fd3e6',1,[],.28);const lab=project(pts[0]);if(lab){ctx.globalAlpha=.6*fog(lab.depth);ctx.fillStyle='#9fd3e6';ctx.font='10px ui-monospace,monospace';ctx.fillText(r/1000+' km',lab.x+4,lab.y-3);ctx.globalAlpha=1;}}}
    const now=P.state.time_s;
    if($('trails').checked)for(const[id,tr]of P.trails){const missile=id.startsWith('m'),span=missile?20:90,points=tr.points.filter(p=>now-p.t<span);
      for(let i=1;i<points.length;i++){const age=(now-points[i].t)/span;line(points[i-1].p,points[i].p,missile?'#fff3d6':colors[tr.team],missile?1.6:1.4,[],(1-age)*(missile?.85:.6));}}
    for(const p of ps)if(p.ghost?.length>1){for(let i=1;i<p.ghost.length;i++)line(p.ghost[i-1],p.ghost[i],colors[p.team],1.2,[5,5],.5);
      const end=project(p.ghost.at(-1));if(end){ctx.globalAlpha=.75*fog(end.depth);ctx.fillStyle=colors[p.team];ctx.font='10px system-ui';ctx.fillText('WT 实际 · '+(p.player?short(p.player):'#'+p.id),end.x+6,end.y-4);ctx.globalAlpha=1;}}
    const dead=new Set(ps.filter(p=>!p.alive).map(p=>p.id));
    for(const m of P.state.missiles){const t=ps.find(p=>p.id===m.target);if(t&&!dead.has(t.id))line(disp(m,dt),disp(t,dt),colors[m.team],1,[2,6],.45);}
    P.screen=[];P.labels=[];
    const sorted=ps.map(p=>{const pos=disp(p,dt);return{p,pos,xy:project(pos)};}).filter(o=>o.xy).sort((a,b)=>b.xy.depth-a.xy.depth);
    for(const{p,pos}of sorted){
      const color=colors[p.team],chosen=picked.has(p.id);
      if(P.view==='3d'&&p.alive){const ground=[pos[0],pos[1],groundAt(pos[0],pos[1])];line(ground,pos,color,1,[2,4],chosen?.6:.3);const g=project(ground);if(g){ctx.globalAlpha=.45*fog(g.depth);ctx.fillStyle='#000';ctx.beginPath();ctx.ellipse(g.x,g.y,Math.max(2,7*ICON),Math.max(.8,2.5*ICON),0,0,Math.PI*2);ctx.fill();ctx.globalAlpha=1;}}
      if(p.order?.destination_m){const dest=[...p.order.destination_m,p.order.altitude_m];line(pos,dest,'#a6ece3',1.5,[6,5]);const d=project(dest);if(d){ctx.strokeStyle='#a6ece3';ctx.lineWidth=1.5;ctx.beginPath();ctx.arc(d.x,d.y,8,0,Math.PI*2);ctx.moveTo(d.x-12,d.y);ctx.lineTo(d.x+12,d.y);ctx.moveTo(d.x,d.y-12);ctx.lineTo(d.x,d.y+12);ctx.stroke();ctx.fillStyle='#a6ece3';ctx.font='11px system-ui';ctx.fillText('目的地',d.x+12,d.y-6);}}
      const xy=drawAircraft(p,pos,chosen);if(!xy)continue;
      ctx.globalAlpha=p.alive?fog(xy.depth)*.4+.6:.4;
      // Labels go far-to-near; one that would cover an earlier label shrinks to its number.
      const lx=xy.x+8,ly=xy.y-10,crowded=!chosen&&P.labels.some(([x,y,w,h])=>lx<x+w&&lx+150>x&&ly-16<y+h&&ly+16>y);
      const tag=p.player?short(p.player):'#'+p.id,head=p.player?tag:'#'+p.id+' '+title(p.aircraft),sub=(p.player?title(p.aircraft)+' · ':'')+(p.alive?fmt(pos[2])+' m · '+fmt(p.speed_mps)+' m/s':'已损失');
      chip(lx,ly,crowded?[tag]:[head,sub],color,chosen||isMe(p));ctx.globalAlpha=1;
      P.screen.push({id:p.id,x:xy.x,y:xy.y});
    }
    for(const m of P.state.missiles){const pos=disp(m,dt),xy=project(pos);if(!xy)continue;const color=colors[m.team];
      if(m.velocity_mps){const v=unit(m.velocity_mps),tail=project(pos.map((x,i)=>x-v[i]*Math.max(3,14*ICON)/xy.scale));if(tail){ctx.strokeStyle='#fff3d6';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(tail.x,tail.y);ctx.lineTo(xy.x,xy.y);ctx.stroke();}}
      ctx.fillStyle='#fffbe8';ctx.beginPath();ctx.arc(xy.x,xy.y,Math.max(.8,2.6*ICON),0,Math.PI*2);ctx.fill();
      ctx.strokeStyle=color;ctx.lineWidth=1;ctx.beginPath();ctx.arc(xy.x,xy.y,Math.max(1.4,(m.seeker?6+2*Math.sin(performance.now()/120):5)*ICON),0,Math.PI*2);ctx.stroke();
      ctx.fillStyle=color;ctx.font='600 10px ui-monospace,monospace';ctx.fillText('M'+m.uid+(m.seeker?' 主动':''),xy.x+5,xy.y-4);}
    if(sel&&sel.alive&&sel.mode==='pilot'&&P.state.status!=='setup')hud({...sel,position_m:disp(sel,dt)},w,h);
    const span=P.view==='2d'?P.map.span:P.camera.radius;$('scale-label').textContent='网格 '+spacing/1000+' km · 视距 '+fmt(span/1000)+' km';
    requestAnimationFrame(draw);
  }
  function fit(){P.camera={x:0,y:0,z:2500,yaw:.45,elevation:.5,radius:mapHalf()*2.4,orbit:0};P.map={x:0,y:0,span:mapHalf()*2.2};$('follow').checked=false;}
  function changeView(view){P.view=view;$('view3d').className=view==='3d'?'active':'';$('view2d').className=view==='2d'?'active':'';$('map-hint').textContent=view==='3d'?'拖动旋转 · Shift + 拖动平移 · 滚轮缩放':'拖动空白处平移 · 滚轮缩放 · 右键下达目的地';canvas.setAttribute('aria-label',view==='3d'?'3D 战场':'俯视地图');}
  function locate(event){const r=canvas.getBoundingClientRect();return{x:event.clientX-r.left,y:event.clientY-r.top};}
  function hit(x,y){return P.screen.filter(p=>Math.hypot(p.x-x,p.y-y)<25).sort((a,b)=>Math.hypot(a.x-x,a.y-y)-Math.hypot(b.x-x,b.y-y))[0];}
  function deploy(team,world){
    if(P.scene.teams[team].length>=P.catalog.max_team){message('每队最多 '+P.catalog.max_team+' 架',true);return;}
    const p=clone(P.scene.teams[team].at(-1));p.position_m[0]=world[0];p.position_m[1]=world[1];P.scene.teams[team].push(p);P.dirty=true;P.pending=null;P.selected=team===0?P.scene.teams[0].length-1:P.scene.teams.flat().length-1;P.formSignature='';render();
  }
  // One aircraft flies to the point; several keep their offsets from the group centre.
  async function destination(world){
    const chosen=selection(),multi=chosen.length>1,group=chosen.filter(p=>p.alive&&!p.tracked&&(!multi||p.mode!=='pilot'));if(!group.length)return;
    const alt=Number($(multi?'g-altitude':'altitude').value),spd=Number($(multi?'g-velocity':'velocity').value);
    const cx=group.reduce((a,p)=>a+p.position_m[0],0)/group.length,cy=group.reduce((a,p)=>a+p.position_m[1],0)/group.length;
    for(const p of group){const d=multi?[world[0]+p.position_m[0]-cx,world[1]+p.position_m[1]-cy]:world.slice(0,2);
      await command('order',{plane:p.id,order:{mode:'manual',destination_m:d,altitude_m:alt,speed_mps:spd}});}
    P.pending=null;P.formSignature='';message((multi?group.length+' 架':'#'+group[0].id)+' 目的地：东 '+fmt(world[0]/1000,1)+' / 北 '+fmt(world[1]/1000,1)+' km'+(multi?'（保持队形）':''));
  }
  canvas.addEventListener('pointerdown',e=>{
    if(e.button!==0)return;canvas.focus();const pos=locate(e),target=hit(pos.x,pos.y);
    if(P.pending){const world=worldAt(pos.x,pos.y,0);if(world){world[0]=clamp(world[0],-mapHalf(),mapHalf());world[1]=clamp(world[1],-mapHalf(),mapHalf());if(P.pending.type==='deploy')deploy(P.pending.team,world);else act(()=>destination(world));}return;}
    if(target)select(target.id,e.ctrlKey||e.metaKey);
    P.drag={...pos,startX:pos.x,startY:pos.y,unit:P.state.status==='setup'&&target?target.id:null,moved:false,pan:e.shiftKey};canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener('pointermove',e=>{
    const pos=locate(e),world=worldAt(pos.x,pos.y,0);if(world)$('coordinates').textContent='东 '+fmt(world[0]/1000,1)+' km · 北 '+fmt(world[1]/1000,1)+' km'+(T.geo&&P.view==='2d'?' · 地形 '+fmt(groundAt(world[0],world[1]))+' m':'');
    const d=P.drag;if(!d)return;
    const dx=pos.x-d.x,dy=pos.y-d.y;d.moved ||= Math.hypot(pos.x-d.startX,pos.y-d.startY)>4;
    if(d.unit!=null&&d.moved){const p=selectedDraft(),point=worldAt(pos.x,pos.y,p.position_m[2]);if(point){p.position_m[0]=clamp(point[0],-P.scene.map_half_m,P.scene.map_half_m);p.position_m[1]=clamp(point[1],-P.scene.map_half_m,P.scene.map_half_m);P.dirty=true;setField('east',fmt(p.position_m[0]/1000,2));setField('north',fmt(p.position_m[1]/1000,2));}}
    else if(P.view==='3d'&&!d.pan){const chase=$('follow').checked&&selected()?.mode==='pilot';if(chase)P.camera.orbit-=dx*.007;else P.camera.yaw-=dx*.007;P.camera.elevation=clamp(P.camera.elevation+dy*.006,-.35,1.48);}
    else {const scale=(P.view==='2d'?P.map.span:P.camera.radius)/Math.min(canvas.clientWidth,canvas.clientHeight);if(P.view==='2d'){P.map.x-=dx*scale;P.map.y+=dy*scale;}else{const b=cameraBasis();P.camera.x-=dx*scale*b.right[0];P.camera.y-=dx*scale*b.right[1];P.camera.x+=dy*scale*b.forward[0];P.camera.y+=dy*scale*b.forward[1];}$('follow').checked=false;}
    d.x=pos.x;d.y=pos.y;
  });
  canvas.addEventListener('pointerup',()=>{P.drag=null;render();});
  canvas.addEventListener('pointercancel',()=>{P.drag=null;});
  canvas.addEventListener('contextmenu',e=>{e.preventDefault();if(P.state.status==='setup'||P.state.status==='playback'||selection().every(p=>p.tracked))return;const pos=locate(e),world=worldAt(pos.x,pos.y,0);if(world)act(()=>destination(world));});
  canvas.addEventListener('wheel',e=>{e.preventDefault();const factor=Math.exp(e.deltaY*.001);if(P.view==='3d')P.camera.radius=clamp(P.camera.radius*factor,600,400000);else P.map.span=clamp(P.map.span*factor,5000,450000);},{passive:false});
  $('terrain').onchange=()=>{T.choice=$('terrain').value;store('wt-sandbox-terrain',T.choice);terrainSync();legendTerrain();};
  $('view3d').onclick=()=>changeView('3d');$('view2d').onclick=()=>changeView('2d');$('fit').onclick=fit;
  $('add-blue').onclick=()=>{P.pending={type:'deploy',team:0};render();message('点击地图放置蓝队飞机');};
  $('add-orange').onclick=()=>{P.pending={type:'deploy',team:1};render();message('点击地图放置橙队飞机');};
  $('preset').onchange=()=>{if(P.state.status!=='setup')return;P.scene=clone(P.catalog.presets[$('preset').value]);P.dirty=true;P.selected=0;P.pending=null;P.formSignature='';$('seed').value=P.scene.seed;$('limit').value=P.scene.time_limit_s;fit();render();};
  for(const id of ['seed','limit'])$(id).onchange=()=>{P.dirty=true;};
  $('aircraft').onchange=()=>{const eq=P.catalog.aircraft.find(a=>a.id===$('aircraft').value),m=eq.missiles.at(-1);fillMissiles(eq.id,m.id);$('ammo').value=m.max;$('chaff').value=eq.chaff_max;editDraft();};
  $('missile').onchange=()=>{const eq=P.catalog.aircraft.find(a=>a.id===$('aircraft').value),m=eq.missiles.find(m=>m.id===$('missile').value);$('ammo').max=m?m.max:0;$('ammo').value=m?m.max:0;editDraft();};
  for(const id of ['ammo','chaff','east','north','archetype','skill','altitude','velocity','heading'])$(id).addEventListener('change',()=>{if(P.state.status==='setup')editDraft();});
  $('remove').onclick=()=>{const p=members()[P.selected];if(P.scene.teams[p.team].length<=1)return;P.scene.teams[p.team].splice(p.slot,1);P.dirty=true;P.selected=0;P.formSignature='';render();};
  $('start').onclick=()=>act(async()=>{if(P.state.status==='playback'){await command('play');return;}if(P.dirty)await configure();await command('start');});
  $('pause').onclick=()=>{releaseKeys();act(()=>command('pause'));};
  $('step').onclick=()=>act(async()=>{if(P.dirty)await configure();await command('step',{seconds:1});});
  $('reset').onclick=()=>{releaseKeys();act(async()=>{await command('reset');P.dirty=false;P.pending=null;P.formSignature='';});};
  $('speed').onchange=()=>{const speed=Number($('speed').value);act(()=>command('speed',{speed}));};
  $('save').onclick=()=>act(async()=>{await configure();const output=await command('save');if(output?.scenario){const a=document.createElement('a');a.href=output.scenario;a.download='scenario.json';a.click();}message('已保存到 '+(output?.path||'outputs/sandbox'));});
  $('apply').onclick=()=>act(async()=>{if(P.state.status==='setup'){editDraft();await configure();message('已应用');}else{releaseKeys();await command('order',{plane:P.selected,order:{mode:'manual',destination_m:null,heading_deg:Number($('heading').value),altitude_m:Number($('altitude').value),speed_mps:Number($('velocity').value)}});P.formSignature='';message('已下达');}});
  $('destination').onclick=()=>{P.pending={type:'destination'};message('点击地图指定目的地，Esc 取消');};
  async function enterPilot(id=P.selected){if(id!==P.selected)select(id);releaseKeys();await command('speed',{speed:1});await command('order',{plane:P.selected,order:{mode:'pilot'}});P.formSignature='';$('follow').checked=true;P.camera.radius=1800;P.camera.elevation=.12;P.camera.orbit=0;const p=selected();P.camera.yaw=-(p.heading_deg||0)*Math.PI/180;changeView('3d');canvas.focus();message('已进入驾驶');}
  $('pilot').onclick=()=>act(async()=>{if(selected().mode==='pilot'){releaseKeys();await command('order',{plane:P.selected,order:{mode:'auto'}});P.formSignature='';}else await enterPilot();});
  $('mode').onchange=()=>{const mode=$('mode').value;act(async()=>{if(mode==='pilot')await enterPilot();else{releaseKeys();await command('order',{plane:P.selected,order:{mode}});P.formSignature='';}});};
  $('fire').onclick=()=>act(()=>command('fire',{plane:P.selected,track:Number($('tracks').value)}));
  $('make-me').onclick=()=>{const p=selected();if(!p)return;if(P.state.status==='playback'){if(!p.player)return;P.me=p.player;store('wt-sandbox-me',P.me);$('me').value=P.me;}else P.ownId=p.id;render();};
  $('own-fire').onclick=()=>act(()=>command('fire',{plane:ownId(),track:Number($('own-tracks').value)}));
  $('own-chaff').onclick=()=>act(()=>command('chaff',{plane:ownId()}));
  $('own-pilot').onclick=()=>act(async()=>{const id=ownId(),p=(P.state.planes||[]).find(q=>q.id===id);if(!p)return;if(p.mode==='pilot'){releaseKeys();await command('order',{plane:id,order:{mode:'auto'}});P.formSignature='';}else{P.ownId=id;await enterPilot(id);}});
  const group=()=>selection().filter(p=>p.alive&&!p.tracked);
  $('g-mode').onchange=()=>{const mode=$('g-mode').value;if(!mode)return;act(async()=>{releaseKeys();for(const p of group())await command('order',{plane:p.id,order:{mode}});P.formSignature='';message(group().length+' 架改为'+controlModes[mode]);});};
  $('g-apply').onclick=()=>act(async()=>{const list=group().filter(p=>p.mode!=='pilot');for(const p of list)await command('order',{plane:p.id,order:{mode:'manual',destination_m:null,heading_deg:Number($('g-heading').value),altitude_m:Number($('g-altitude').value),speed_mps:Number($('g-velocity').value)}});P.formSignature='';message('已向 '+list.length+' 架下令');});
  $('g-destination').onclick=()=>{P.pending={type:'destination'};message('点击地图指定目的地（保持队形），Esc 取消');};
  $('g-clear').onclick=()=>select(P.selected);
  $('drop-chaff').onclick=()=>act(()=>command('chaff',{plane:P.selected}));
  async function loadReplays(){try{P.replays=await api('/api/replays');}catch(error){message('无法读取回放列表：'+error.message,true);}render();}
  $('wrpl-pick').onclick=()=>$('wrpl-file').click();
  $('wrpl-file').onchange=()=>{const files=[...$('wrpl-file').files];$('wrpl-file').value='';importFiles(files);};
  $('wrpl-game-ok').onclick=()=>act(async()=>{await command('wrpl_game',{path:$('wrpl-game-path').value});message('已设置游戏目录，继续导入');});
  $('wrpl-game-path').addEventListener('keydown',e=>{if(e.key==='Enter')$('wrpl-game-ok').click();});
  $('wrpl-dl-go').onclick=()=>act(async()=>{await command('wrpl_download');message('正在下载解析后端');});
  const dropZone=$('replay-side'),hasFiles=e=>[...(e.dataTransfer?.types||[])].includes('Files');
  dropZone.addEventListener('dragenter',e=>{if(!hasFiles(e))return;e.preventDefault();P.dragDepth++;dropZone.classList.add('drop');});
  dropZone.addEventListener('dragover',e=>{if(!hasFiles(e))return;e.preventDefault();e.dataTransfer.dropEffect=P.state?.fork?'none':'copy';});
  dropZone.addEventListener('dragleave',()=>{P.dragDepth=Math.max(0,P.dragDepth-1);if(!P.dragDepth)dropZone.classList.remove('drop');});
  dropZone.addEventListener('drop',e=>{if(!hasFiles(e))return;e.preventDefault();P.dragDepth=0;dropZone.classList.remove('drop');if(!P.state?.fork)importFiles(e.dataTransfer.files);});
  // A file dropped elsewhere must not make the browser leave the page.
  window.addEventListener('dragover',e=>{if(hasFiles(e))e.preventDefault();});
  window.addEventListener('drop',e=>{if(hasFiles(e))e.preventDefault();});
  $('mode-replay').onclick=()=>{if(P.mode==='replay')return;if(P.state.status!=='setup'){message('先重置推演再切到回放',true);return;}P.mode='replay';P.selected=-1;P.pending=null;loadReplays();render();};
  $('mode-setup').onclick=()=>{if(P.mode==='setup')return;releaseKeys();act(async()=>{if(P.state.replay_file||inReplay())await command('unload');P.mode='setup';P.selected=0;P.formSignature='';$('fork-panel').hidden=true;fit();});};
  $('me').onchange=()=>{P.me=$('me').value;store('wt-sandbox-me',P.me);render();};
  $('show-ai').onchange=()=>{P.showAI=$('show-ai').checked;render();};
  $('tl-range').addEventListener('input',()=>{P.seeking=true;$('tl-label').textContent=clockText(Number($('tl-range').value))+' / '+clockText(P.state.replay?.duration_s||0);});
  $('tl-range').addEventListener('change',()=>{const t=Number($('tl-range').value);act(async()=>{try{await command('seek',{t});}finally{P.seeking=false;}});});
  $('fork').onclick=openFork;$('fork-cancel').onclick=()=>{$('fork-panel').hidden=true;};
  $('fork-go').onclick=()=>act(async()=>{
    const s=P.state,control={};for(const sel of $('fork-rows').querySelectorAll('select'))if(sel.value!=='track')control[sel.dataset.id]=sel.value;
    const me=s.planes.find(p=>isMe(p)),includeAI=$('fork-ai').checked;$('fork-panel').hidden=true;
    message('正在建立推演…');
    const r=await command('fork',{t:s.time_s,control,include_ai:includeAI,me:me?me.id:null},2500);
    if(includeAI)P.showAI=true;
    if(r&&r.me!=null){P.selected=r.me;P.formSignature='';if(control[me.id]==='pilot'){$('follow').checked=true;P.camera.radius=1800;P.camera.elevation=.12;P.camera.orbit=0;P.camera.yaw=-(me.heading_deg||0)*Math.PI/180;changeView('3d');canvas.focus();}}
    message('接管就绪 · '+r.planes+' 架'+(r.t0<r.t_fork-1e-6?'，快进到 '+clockText(r.t_fork)+' 后暂停':'')+'。点「继续」开始。');});

  // Inputs are bound to the canvas, never text fields. A short server lease also
  // neutralizes held controls if the browser disappears without sending keyup.
  const controlKeys=new Set(['KeyW','KeyS','KeyA','KeyD','ArrowUp','ArrowDown','ArrowLeft','ArrowRight','Equal','Minus','NumpadAdd','NumpadSubtract','KeyR','KeyF','KeyL','Space']);
  function input(){const has=(...codes)=>codes.some(c=>P.keys.has(c));return{roll:Number(has('KeyD','ArrowRight'))-Number(has('KeyA','ArrowLeft')),pitch:Number(has('KeyW','ArrowUp'))-Number(has('KeyS','ArrowDown')),throttle:Number(has('Equal','NumpadAdd','KeyR'))-Number(has('Minus','NumpadSubtract','KeyF')),airbrake:has('Space'),level:has('KeyL')};}
  async function sendInput(){
    const p=P.state&&selected();if(!p||p.mode!=='pilot'||!p.alive||P.inputPending)return;
    // A key change that arrives while a send is in flight is sent right after it, never dropped.
    P.inputPending=true;let keys;
    try{do{keys=input();await command('keys',{plane:p.id,keys});}while(JSON.stringify(input())!==JSON.stringify(keys));}
    catch(error){if(P.keys.size)message(error.message,true);}finally{P.inputPending=false;}
  }
  function releaseKeys(){
    P.keys.clear();
    const p=P.state&&selected();
    if(p?.mode==='pilot'&&P.catalog) {
      // Neutral release is never suppressed by a pending heartbeat request.
      command('keys',{plane:p.id,keys:{roll:0,pitch:0,throttle:0,airbrake:false,level:false}}).catch(()=>{});
    }
  }
  canvas.addEventListener('keydown',e=>{if(e.code==='Escape'){P.pending=null;$('fork-panel').hidden=true;releaseKeys();hideMessage();render();return;}if(e.code==='KeyC'){P.camera.orbit=0;P.camera.elevation=.12;return;}if(!controlKeys.has(e.code)||selected()?.mode!=='pilot'||P.state.status!=='running')return;e.preventDefault();if(!e.repeat){P.keys.add(e.code);sendInput();}});
  window.addEventListener('keyup',e=>{if(P.keys.delete(e.code)){if(document.activeElement===canvas)e.preventDefault();sendInput();}});
  canvas.addEventListener('blur',()=>{releaseKeys();render();});window.addEventListener('blur',releaseKeys);
  document.addEventListener('visibilitychange',()=>{if(document.hidden)releaseKeys();});
  window.addEventListener('keydown',e=>{if(e.code==='Escape'){P.pending=null;$('fork-panel').hidden=true;releaseKeys();render();}});
  setInterval(()=>{if(P.state?.status==='running'&&selected()?.mode==='pilot'&&document.activeElement===canvas)sendInput();},120);

  async function init(){
    try{P.catalog=await api('/api/catalog');for(const speed of P.catalog.speeds)option($('speed'),speed,speed+'×');const groups=new Map();for(const a of P.catalog.aircraft){const g=family(a.id);if(!groups.has(g))groups.set(g,[]);groups.get(g).push(a);}
      for(const g of [...groups.keys()].sort((a,b)=>(a==='其他')-(b==='其他')||groups.get(b).length-groups.get(a).length)){const og=document.createElement('optgroup');og.label=g+' · '+groups.get(g).length;
        for(const a of groups.get(g).sort((x,y)=>title(x.id).localeCompare(title(y.id),'zh')))option(og,a.id,title(a.id)+(names[a.id]&&family(a.id)!=='中国'?'（'+names[a.id][1]+'）':'')+' — '+a.missiles.map(m=>missileName(m.id)).join(' / '));$('aircraft').append(og);}for(const a of P.catalog.archetypes)option($('archetype'),a,archetypes[a]||a);for(const s of P.catalog.skills)option($('skill'),s,s==='top'?'高手':'普通');try{T.maps=(await api('/api/terrains')).maps;}catch(error){T.maps=[];}fillTerrainPicker();await poll();if(P.mode==='replay')await loadReplays();$('preset').value=Object.keys(P.catalog.presets).find(k=>P.catalog.presets[k].name===P.scene.name)||'duel';$('seed').value=P.scene.seed;$('limit').value=P.scene.time_limit_s;fit();draw();
      while(true){await sleep(50);try{await poll();}catch(error){message('连接中断：'+error.message,true);await sleep(1200);}}
    }catch(error){message('启动失败：'+error.message,true);}
  }
  init();
})();
