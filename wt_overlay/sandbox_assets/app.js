'use strict';
(() => {
  const $ = id => document.getElementById(id);
  const canvas = $('map'), ctx = canvas.getContext('2d');
  const P = {catalog: null, state: null, scene: null, dirty: false, selected: 0, busy: false,
    view: '3d', pending: null, camera: {x: 0, y: 0, z: 2500, yaw: .45, elevation: .62, radius: 120000, orbit: 0}, received: 0,
    map: {x: 0, y: 0, span: 140000}, screen: [], trails: new Map(), events: new Map(), keys: new Set(),
    drag: null, lastTime: -1, lastStatus: '', formSignature: '', inputPending: false,
    mode: 'setup', replays: [], me: '', showAI: false, seeking: false, markerFile: null, meFile: null, lastFork: false};
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
  function isMe(p){ const s=P.state; if(!s)return false; if(s.fork)return p.id===s.fork.me; return s.status==='playback'&&!!P.me&&p.player===P.me; }
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
    throw new Error('命令仍在处理，请查看当前状态');
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
      $('chaff').value=draft.chaff ?? ''; $('chaff').placeholder='按种子抽样';
      $('east').value=fmt(draft.position_m[0]/1000,2);$('north').value=fmt(draft.position_m[1]/1000,2);
      $('archetype').value=draft.archetype||'middle';$('skill').value=draft.skill||'top';
    }
    $('mode').value=p.mode;
    const order=p.order;
    $('altitude').value=fmt(order?order.altitude_m:p.position_m[2]);
    $('velocity').value=fmt(order?order.speed_mps:p.speed_mps);
    $('heading').value=fmt(order?order.heading_deg:p.heading_deg,1);
  }
  function select(id) {releaseKeys();P.selected=id;P.formSignature='';render();}
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
  function render() {
    if(!P.state)return;
    const s=P.state, setup=s.status==='setup', done=s.status==='completed', running=s.status==='running';
    const playback=s.status==='playback', playing=!!s.replay?.playing, replayUI=P.mode==='replay';
    const status={setup:'准备布置',running:'推演中',paused:'已暂停',stepping:'单步推进中',completed:'推演结束',error:'运行错误',playback:playing?'WT 回放 · 播放中':'WT 回放 · 已暂停'};
    $('status').textContent=(s.fork?'接管推演 · ':'')+status[s.status];$('setup-tag').textContent=setup?'可部署':'锁定布置';
    $('clock').textContent=clockText(s.time_s+(s.fork?.offset||0));
    $('speed').value=s.speed;$('actual').textContent='实际 '+fmt(s.actual_speed,1)+'×';
    $('start').textContent=playback?(playing?'▶ 播放中':'▶ 播放'):setup?'▶ 开始推演':running?'▶ 推演中':'▶ 继续推演';
    $('start').disabled=P.busy||running||done||s.status==='error'||s.status==='stepping'||playback&&playing||setup&&replayUI;
    $('pause').disabled=P.busy||!(['running','stepping'].includes(s.status)||playback&&playing);
    $('step').disabled=P.busy||!['setup','paused'].includes(s.status)||setup&&replayUI;
    $('reset').disabled=P.busy||setup||playback;$('reset').textContent=s.fork?'↺ 返回回放':'↺ 重置布置';$('preset').disabled=!setup||P.busy;
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
      const list=$(team?'orange-list':'blue-list');list.replaceChildren();
      for(const p of ps.filter(p=>p.team===team)) {
        const row=document.createElement('button');row.className='unit-row'+(team?' orange-row':'')+(p.id===P.selected?' selected':'')+(!p.alive?' dead':'');
        const name=document.createElement('span');name.className='row-title';name.textContent=p.player?short(p.player)+' · '+title(p.aircraft):'#'+p.id+' '+title(p.aircraft);
        const meta=document.createElement('span');meta.className='row-meta';meta.textContent=(p.missile?missileName(p.missile)+' · ':'')+(p.alive?fmt(p.position_m[2])+' m · '+fmt(p.speed_mps)+' m/s · '+(phases[p.phase]||p.phase||'接敌'):'已损失');
        row.append(name,meta);row.onclick=()=>select(p.id);list.append(row);
      }
    }
    const p=selected();$('empty-selection').hidden=!!p;$('unit-panel').hidden=!p;
    if(p) {
      $('unit-id').textContent=teams[p.team]+' #'+p.id;$('unit-name').textContent=p.player?short(p.player)+' · '+title(p.aircraft):title(p.aircraft);$('unit-name').className='unit-name '+(p.team?'orange':'blue');
      $('unit-state').replaceChildren();
      for(const text of ['高度 '+fmt(p.position_m[2])+' m','速度 '+fmt(p.speed_mps)+' m/s','航向 '+fmt(p.heading_deg,1)+'°','导弹 '+fmt(p.missiles)+' 枚',p.chaff==null?(playback?'箔条 未知':'箔条 按种子抽样'):'箔条 '+fmt(p.chaff)+' 束',p.alive?'存活':'已损失']) {const t=document.createElement('span');t.textContent=text;$('unit-state').append(t);}
      const locked=playback||!!p.tracked;
      $('edit-panel').hidden=!setup;$('mode-panel').hidden=setup||locked;$('nav-fields').hidden=$('apply').hidden=locked;$('track-note').hidden=!locked;
      if(locked)$('track-note').textContent=playback?'原样回放：位置与速度来自回放，姿态为估算（俯仰取速度方向，滚转按协调转弯）。'+(p.controllable?'可接管'+(p.fork_missile?'，携带 '+missileName(p.fork_missile):'，无已建模导弹')+'。':'不能接管：'+(p.control_reason||'—')+'。'):'钉在 WT 轨迹上：飞行、发射与箔条按回放记录'+(p.release_at!=null?'；回放 '+fmt(p.release_at,2)+' s 起接管':'')+'。';
      $('remove').hidden=!setup;$('remove').disabled=P.busy||setup&&count[p.team]<=1;
      $('apply').textContent=setup?'应用初始状态':'下达航向 / 高度 / 速度';
      $('apply').disabled=P.busy||!p.alive||done||s.status==='error'||p.mode==='pilot';
      $('destination').hidden=$('pilot').hidden=setup||locked;
      $('destination').disabled=$('pilot').disabled=P.busy||!p.alive||done||s.status==='error';
      $('pilot').textContent=p.mode==='pilot'?'退出驾驶 → 自主脚本':'进入键盘驾驶（1×）';
      $('mode').disabled=P.busy||!p.alive||done||s.status==='error';
      $('mode').value=p.mode;
      $('mode-help').textContent=p.mode==='auto'?'飞行与武器由原脚本决定，观测限制保持生效。':p.mode==='pilot'?'点地图获得焦点；W/S 俯仰，A/D 滚转，R/F 油门，空格减速板，按住 L 自动改平；拖动环视，C 回正。暂停或失焦释放按键。':'设置航向或地图目的地；高度和速度由真实飞行模型跟随。雷达自动搜索，武器由你发射。';
      const signature=p.id+'|'+setup+'|'+p.mode;
      if(P.formSignature!==signature){rebuildForm();P.formSignature=signature;}
      $('weapon-panel').hidden=setup||locked||p.mode==='auto';
      const previous=$('tracks').value;$('tracks').replaceChildren();
      for(const c of p.tracks)option($('tracks'),c.track,'航迹 '+c.track+' · '+fmt(c.range_m/1000,1)+' km · '+fmt(c.bearing_deg)+'°');
      if(p.tracks.length===0)option($('tracks'),'','暂无可发射雷达航迹');
      else if(p.tracks.some(c=>String(c.track)===previous))$('tracks').value=previous;
      $('fire').disabled=P.busy||done||!p.alive||!p.tracks.length||p.missiles<=0;
      $('drop-chaff').disabled=P.busy||done||!p.alive||p.chaff<=0;
      $('control-banner').hidden=p.mode!=='pilot'||!p.alive||done;
      if(p.mode==='pilot')$('control-banner').textContent=(document.activeElement===canvas?'驾驶焦点已启用':'点击战场启用驾驶键盘')+' · W/S 拉起/下压 · A/D 滚转 · R/F 油门 · 空格减速板 · L 改平 · 拖动环视 / C 回正';
    } else $('control-banner').hidden=true;
    $('deployment-help').textContent=P.pending?.type==='deploy'?'在地图点击部署'+teams[P.pending.team]+'飞机。按 Esc 取消。':setup?'点选飞机编辑参数；拖动飞机修改位置。点击「部署」后，在地图选位置。':'点选飞机下令或进入驾驶。重置后可重新布置，当前未完成对局保存为中止。';
    $('health').textContent='导弹错误 '+s.missile_errors+' · 飞行模型越界 '+s.fm_faults;
    $('health').style.color=s.missile_errors||s.fm_faults?'#ff979a':'';
    $('terminal').hidden=!done;
    if(done){$('terminal').replaceChildren();const strong=document.createElement('strong');strong.textContent=s.output?.winner_team==null?'推演结束 · 平局':teams[s.output.winner_team]+'获胜';const sub=document.createElement('div');sub.textContent=(reasons[s.reason]||s.reason)+' · '+fmt(s.time_s,2)+' s';$('terminal').append(strong,sub);}
    $('run-note').textContent=playback?'原样回放：不经过任何模型；暂停后可从此刻接管。':s.fork&&running?'接管推演：钉轨飞机按 WT 记录，已接管飞机由现有模型推进。':running?'倍率为期望速度；实际倍率反映当前计算能力。':setup?'布置完成后开始；也可单步建立暂停的对局。':done?'已停止推进；场景、结果与完整回放已保存。':'物理时钟已暂停，可以调整所选飞机的命令。';
    if(s.error)message(s.error,true);
    if(s.output){const o=$('outputs');o.replaceChildren();const t=document.createElement('p');t.textContent='记录：'+({completed:'已完成',partial:'尚未结束',cancelled:'已中止',server_stopped:'服务关闭',error:'运行错误'}[s.output.status]||s.output.status);o.append(t);for(const [file,url]of Object.entries(s.output.links)){const a=document.createElement('a');a.href=url;a.textContent=file==='result.json'?'下载结果 JSON':file==='scenario.json'?'下载场景 JSON':'下载完整回放 JSONL';o.append(a);}}
    renderEvents();
  }
  function renderReplay(){
    const s=P.state,list=$('replay-list');list.replaceChildren();
    if(!P.replays.length){const e=document.createElement('p');e.className='help';e.textContent='没有找到回放（outputs/engagements/wt_real 与 outputs/sandbox）。';list.append(e);}
    for(const r of P.replays){const row=document.createElement('button');row.className='unit-row replay-row'+(s.replay_file===r.file?' selected':'');
      const name=document.createElement('span');name.className='row-title';name.textContent=(r.kind==='wt'?'WT · ':'沙盘 · ')+(r.kind==='wt'?r.file.split('/').at(-1).replace(/\.jsonl$/,''):r.file.split('/').at(-2));
      const meta=document.createElement('span');meta.className='row-meta';meta.textContent=new Date(r.mtime*1000).toLocaleString('zh-CN',{hour12:false})+' · 时长 '+(r.duration_s==null?'—':clockText(r.duration_s))+' · 玩家 '+r.players+(r.units>r.players?' · AI '+(r.units-r.players):'');
      row.append(name,meta);row.disabled=P.busy||!!s.fork;row.onclick=()=>act(async()=>{await command('load_replay',{file:r.file});});list.append(row);}
    $('replay-tag').textContent=!s.replay_file?'未加载':s.fork?'接管推演':s.replay?.playing?'播放中':'已暂停';
    const ro=s.replay?.roster;
    if(ro&&P.meFile!==s.replay.file){P.meFile=s.replay.file;const sel=$('me');sel.replaceChildren();option(sel,'','（未选择）');
      const players=Object.values(ro).filter(r=>!r.ai).sort((a,b)=>a.team-b.team||a.player.localeCompare(b.player));
      for(const r of players)option(sel,r.player,short(r.player)+' · '+title(r.aircraft)+' · '+teams[r.team]);
      const saved=recall('wt-sandbox-me');P.me=players.some(r=>r.player===saved)?saved:'';sel.value=P.me;}
    $('me').disabled=!ro||P.busy;$('show-ai').checked=P.showAI;
    $('fork').disabled=P.busy||s.status!=='playback'||!!s.replay?.playing;
    $('fork').title=s.replay?.playing?'先暂停在要接管的时刻':'';
    const box=$('fork-info');box.hidden=!s.fork;
    if(s.fork){const f=s.fork;box.replaceChildren();const h=document.createElement('strong');h.textContent='WT 回放接管 · '+f.file;box.append(h);
      for(const text of ['接管时刻 '+clockText(f.t_fork)+' · 世界起点 '+clockText(f.t0),'接管：'+(f.released.length?f.released.map(r=>short(r.player)+'（'+(controlModes[r.mode]||r.mode)+'）').join('、'):'无（全部按 WT 轨迹）'),'未重飞的回放发射 '+f.skipped+' 枚（事件栏有原因）',...(f.notes||[])]){const e=document.createElement('p');e.textContent=text;box.append(e);}}
    $('timeline').hidden=!s.replay;
    if(s.replay){const r=s.replay,range=$('tl-range');range.min=r.start_s;range.max=r.duration_s;if(!P.seeking)range.value=s.time_s;range.disabled=P.busy;
      $('tl-label').textContent=clockText(P.seeking?Number(range.value):s.time_s)+' / '+clockText(r.duration_s);
      if(P.markerFile!==r.file){P.markerFile=r.file;const marks=$('tl-markers'),span=Math.max(1e-6,r.duration_s-r.start_s);marks.replaceChildren();
        for(const m of r.markers){const e=document.createElement('span');e.className='tl-mark '+m.kind;e.textContent=m.kind==='launch'?'▲':'✕';e.style.left=clamp((m.t-r.start_s)/span*100,0,100)+'%';e.title=clockText(m.t)+' · '+m.text;marks.append(e);}}}
  }
  function openFork(){
    const s=P.state;if(s.status!=='playback'||s.replay?.playing)return;
    $('fork-title').textContent='从 '+clockText(s.time_s)+' 接管';const rows=$('fork-rows');rows.replaceChildren();
    for(const p of s.planes.filter(p=>p.alive&&!p.ai).sort((a,b)=>a.team-b.team||a.id-b.id)){
      const row=document.createElement('div');row.className='fork-row';const label=document.createElement('div');
      const n=document.createElement('span');n.className=p.team?'orange':'blue';n.textContent=short(p.player)+(isMe(p)?'（我）':'');
      const sub=document.createElement('small');sub.textContent=title(p.aircraft)+' · '+(p.controllable?(p.fork_missile?missileName(p.fork_missile)+' × '+fmt(p.missiles):'无已建模导弹'):'只能 WT 轨迹：'+p.control_reason);label.append(n,sub);
      const sel=document.createElement('select');sel.dataset.id=p.id;for(const [k,v] of Object.entries(controlModes))option(sel,k,v);
      sel.value=p.controllable&&isMe(p)?'pilot':'track';sel.disabled=!p.controllable;row.append(label,sel);rows.append(row);}
    if(!rows.children.length){const e=document.createElement('p');e.className='help';e.textContent='此刻没有存活的玩家飞机。';rows.append(e);}
    $('fork-ai').checked=P.showAI;$('fork-panel').hidden=false;
  }
  function eventText(e) {
    const plane=e.plane??e.shooter, who=plane==null?'':nameOf(plane)+' ';
    if(e.kind==='launch')return who+'发射'+(inReplay()&&e.missile?' '+missileName(baseMissile(e.missile)):'导弹')+' #'+e.uid+' → '+nameOf(e.target)+(e.target_basis?'（目标依据：'+(e.target_basis.source||e.target_basis)+'）':'');
    if(e.kind==='missile_end'&&e.outcome)return '导弹 #'+e.uid+' 结束 · '+(outcomes[e.outcome]||e.outcome)+'（依据：'+(e.outcome_basis||'未知')+'）';
    if(e.kind==='kill'&&inReplay())return nameOf(e.killer)+' 击落 '+nameOf(e.victim)+(e.source?'（'+e.source+'）':'');
    if(e.kind==='replay_shot_skipped')return '回放发射未重飞 · '+missileName(baseMissile(e.missile))+' · '+e.reason;
    if(e.kind==='replay_release')return who+'接管 · '+(controlModes[e.mode]||e.mode);
    if(e.kind==='rwr')return who+(e.warning==='missile'?'导弹告警':'雷达锁定告警');
    if(e.kind==='datalink_lost')return '导弹 #'+e.uid+' 支援中断 · '+({track_lost:'载机丢失航迹',shooter_dead:'载机已损失',seeker_track:'导引头已截获'}[e.reason]||e.reason||'见回放');
    if(e.kind==='seeker_on')return '导弹 #'+e.uid+' 导引头开启';
    if(e.kind==='missile_end')return '导弹 #'+e.uid+' 终止 · '+({lifetime:'达到仿真寿命上限',fuse:'近炸触发',ground:'触地',error:'模型错误',target_dead:'目标已损失'}[e.result]||e.result)+(['fuse','error'].includes(e.result)?'':'（额外脱靶原因未单独记录）');
    if(e.kind==='death')return nameOf(e.plane)+' 损失 · '+({missile:'导弹击落',crash:'坠毁',out_of_bounds:'出界',left_replay:'离开回放'}[e.cause]||(String(e.cause).startsWith('replay_')?'回放记录 · '+e.cause.slice(7):e.cause));
    if(e.kind==='chaff')return who+'投放箔条';
    if(e.kind==='phase'&&['evade','recommit','round2'].includes(e.to))return who+(phases[e.to]||e.to);
    if(e.kind==='human_order')return who+({auto:'返回自主脚本',manual:'接收人工导航',pilot:'进入玩家驾驶'}[e.mode]);
    if(e.kind==='human_input')return who+'操纵输入 · 滚转 '+e.keys.roll+' / 俯仰 '+e.keys.pitch;
    if(e.kind==='waypoint_arrived')return who+'到达目的地区域，保持当前航向';
    if(e.kind==='observed_contact'&&e.source==='radar'&&e.change==='appeared')return who+'雷达观测出现';
    if(e.kind==='end')return '推演结束 · '+(reasons[e.reason]||e.reason);
    if(e.kind==='missile_error')return '导弹模型错误 · '+(e.error||'查看完整回放');
    return null;
  }
  function renderEvents(){
    const s=P.state,start=s.event_total-s.events.length;
    s.events.forEach((e,i)=>{const t=eventText(e);if(t)P.events.set(start+i,{...e,text:t});});
    if(P.events.size>100)for(const k of [...P.events.keys()].slice(0,P.events.size-100))P.events.delete(k);
    $('event-count').textContent=P.events.size+' 条';
    const list=$('events');list.replaceChildren();
    if(!P.events.size){const p=document.createElement('p');p.className='help';p.textContent='发现、发射、告警、支援中断和结果将在这里出现。';list.append(p);}
    for(const e of [...P.events.values()].reverse()) {const row=document.createElement('div');row.className='event '+(e.kind==='death'||e.kind==='kill'?'hit':e.kind==='rwr'?'warning':e.kind.startsWith('human')?'order':'');const t=document.createElement('time');t.textContent=fmt(e.t+(P.state.fork?.offset||0),1);const text=document.createElement('span');text.textContent=e.text;row.append(t,text);list.append(row);}
  }
  async function poll(){
    const s=await api('/api/state');
    if(!P.scene||(!P.dirty&&s.status==='setup'))P.scene=clone(s.scenario);
    if(s.status==='playback'||s.fork)P.mode='replay';
    const fork=!!s.fork;if(fork!==P.lastFork||(s.replay_file||null)!==(P.state?.replay_file||null)){P.trails.clear();P.events.clear();P.lastTime=-1;P.lastFork=fork;}
    if(s.time_s<P.lastTime||s.status==='setup'&&P.lastStatus!=='setup'){P.trails.clear();P.events.clear();P.lastTime=-1;P.formSignature='';}
    if(s.time_s>P.lastTime){
      for(const p of s.planes.filter(p=>p.alive&&(P.showAI||!p.ai)))trail('p'+p.id,p.position_m,s.time_s,p.team);
      for(const m of s.missiles)trail('m'+m.uid,m.position_m,s.time_s,m.team);
      P.lastTime=s.time_s;
    }
    if(s.status!=='running'&&P.lastStatus==='running')releaseKeys();
    P.lastStatus=s.status;P.state=s;P.received=performance.now();render();
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
  function bodyAxes(p){
    const H=p.heading_deg*Math.PI/180,Pt=(p.pitch_deg||0)*Math.PI/180,R=(p.roll_deg||0)*Math.PI/180;
    const f=[Math.sin(H)*Math.cos(Pt),Math.cos(H)*Math.cos(Pt),Math.sin(Pt)],r0=[Math.cos(H),-Math.sin(H),0],u0=cross(r0,f);
    const r=r0.map((v,i)=>v*Math.cos(R)-u0[i]*Math.sin(R));return{f,r,u:cross(r,f)};
  }
  const HULL=[[1.35,0],[.55,.1],[.05,.22],[-.15,.95],[-.42,.95],[-.38,.2],[-.72,.22],[-.78,.5],[-.98,.5],[-.92,0]];
  function drawAircraft(p,pos,chosen){
    const xy=project(pos);if(!xy)return null;
    const color=colors[p.team],ax=bodyAxes(P.view==='2d'?{heading_deg:p.heading_deg}:p),px=chosen&&p.mode==='pilot'&&$('follow').checked&&P.view==='3d'?44:p.ai?11:17,size=clamp(px/xy.scale,12,6000);
    const at=(x,y,z=0)=>pos.map((v,i)=>v+(ax.f[i]*x+ax.r[i]*y+ax.u[i]*z)*size);
    const outline=[...HULL,...HULL.slice(1,-1).reverse().map(([x,y])=>[x,-y])].map(([x,y])=>at(x,y));
    const poly=clipPoly(outline),fin=clipPoly([at(-.45,0),at(-.9,0,.42),at(-.98,0,.42),at(-.92,0)]);
    const toEye=P.view==='2d'?[0,0,1]:unit(B.eye.map((v,i)=>v-pos[i])),top=dot(ax.u,toEye)>=0;
    const me=isMe(p);ctx.globalAlpha=(p.alive?1:.3)*(p.ai?.55:1);
    if(chosen){ctx.shadowColor=color;ctx.shadowBlur=14;}
    if(poly){ctx.fillStyle=top?color:shade(color,.55);ctx.strokeStyle=chosen||me?'#ffffff':shade(color,1.35);ctx.lineWidth=me?2.4:chosen?1.6:1;ctx.lineJoin='round';
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
    const half=mapHalf(),ps=planes(),sel=selected(),dt=lead();
    if($('follow').checked&&sel){
      const at=disp(sel,dt);P.camera.x=at[0];P.camera.y=at[1];P.camera.z=at[2];P.map.x=at[0];P.map.y=at[1];
      if(sel.mode==='pilot'&&P.view==='3d'){const target=-sel.heading_deg*Math.PI/180+P.camera.orbit;let d=target-P.camera.yaw;d=Math.atan2(Math.sin(d),Math.cos(d));P.camera.yaw+=d*.12;}
    }
    B=P.view==='3d'?cameraBasis():null;
    backdrop(w,h);
    const field=clipPoly([[-half,-half,0],[half,-half,0],[half,half,0],[-half,half,0]]);
    if(field){ctx.fillStyle=P.view==='3d'?'rgba(70,120,140,.07)':'rgba(70,120,140,.06)';ctx.beginPath();field.forEach((q,i)=>i?ctx.lineTo(q.x,q.y):ctx.moveTo(q.x,q.y));ctx.closePath();ctx.fill();}
    const spacing=P.view==='2d'&&P.map.span<40000?5000:10000;
    for(let i=Math.ceil(-half/spacing)*spacing;i<=half;i+=spacing){const c=i===0?'#5d8aa3':'#36586b';fadedLine([i,-half,0],[i,half,0],c,1,8,.75);fadedLine([-half,i,0],[half,i,0],c,1,8,.75);}
    for(const[a,b]of [[[ -half,-half,0],[half,-half,0]],[[half,-half,0],[half,half,0]],[[half,half,0],[-half,half,0]],[[-half,half,0],[-half,-half,0]]])fadedLine(a,b,'#6f9bb3',1.6,8);
    for(const [axis,p]of [['N',[0,half*1.06,0]],['S',[0,-half*1.06,0]],['E',[half*1.06,0,0]],['W',[-half*1.06,0,0]]]){const xy=project(p);if(xy){ctx.globalAlpha=fog(xy.depth);ctx.fillStyle='#8fbad1';ctx.font='600 12px system-ui';ctx.textAlign='center';ctx.fillText(axis,xy.x,xy.y+4);ctx.textAlign='left';ctx.globalAlpha=1;}}
    if(sel&&sel.alive){const c=disp(sel,dt);for(const r of [10000,20000,40000]){const pts=[];for(let k=0;k<=72;k++){const a=k/72*Math.PI*2;pts.push([c[0]+Math.sin(a)*r,c[1]+Math.cos(a)*r,0]);}
      for(let k=1;k<pts.length;k++)line(pts[k-1],pts[k],'#9fd3e6',1,[],.28);const lab=project([c[0],c[1]+r,0]);if(lab){ctx.globalAlpha=.6*fog(lab.depth);ctx.fillStyle='#9fd3e6';ctx.font='10px ui-monospace,monospace';ctx.fillText(r/1000+' km',lab.x+4,lab.y-3);ctx.globalAlpha=1;}}}
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
      const color=colors[p.team],chosen=p.id===P.selected;
      if(P.view==='3d'&&p.alive){const ground=[pos[0],pos[1],0];line(ground,pos,color,1,[2,4],chosen?.6:.3);const g=project(ground);if(g){ctx.globalAlpha=.45*fog(g.depth);ctx.fillStyle='#000';ctx.beginPath();ctx.ellipse(g.x,g.y,7,2.5,0,0,Math.PI*2);ctx.fill();ctx.globalAlpha=1;}}
      if(p.order?.destination_m){const dest=[...p.order.destination_m,p.order.altitude_m];line(pos,dest,'#a6ece3',1.5,[6,5]);const d=project(dest);if(d){ctx.strokeStyle='#a6ece3';ctx.lineWidth=1.5;ctx.beginPath();ctx.arc(d.x,d.y,8,0,Math.PI*2);ctx.moveTo(d.x-12,d.y);ctx.lineTo(d.x+12,d.y);ctx.moveTo(d.x,d.y-12);ctx.lineTo(d.x,d.y+12);ctx.stroke();ctx.fillStyle='#a6ece3';ctx.font='11px system-ui';ctx.fillText('目的地',d.x+12,d.y-6);}}
      const xy=drawAircraft(p,pos,chosen);if(!xy)continue;
      ctx.globalAlpha=p.alive?fog(xy.depth)*.4+.6:.4;
      // Labels go far-to-near; one that would cover an earlier label shrinks to its number.
      const lx=xy.x+16,ly=xy.y-14,crowded=!chosen&&P.labels.some(([x,y,w,h])=>lx<x+w&&lx+150>x&&ly-16<y+h&&ly+16>y);
      const tag=p.player?short(p.player):'#'+p.id,head=p.player?tag:'#'+p.id+' '+title(p.aircraft),sub=(p.player?title(p.aircraft)+' · ':'')+(p.alive?fmt(pos[2])+' m · '+fmt(p.speed_mps)+' m/s':'已损失');
      chip(lx,ly,crowded?[tag]:[head,sub],color,chosen||isMe(p));ctx.globalAlpha=1;
      P.screen.push({id:p.id,x:xy.x,y:xy.y});
    }
    for(const m of P.state.missiles){const pos=disp(m,dt),xy=project(pos);if(!xy)continue;const color=colors[m.team];
      if(m.velocity_mps){const v=unit(m.velocity_mps),tail=project(pos.map((x,i)=>x-v[i]*14/xy.scale));if(tail){ctx.strokeStyle='#fff3d6';ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(tail.x,tail.y);ctx.lineTo(xy.x,xy.y);ctx.stroke();}}
      ctx.fillStyle='#fffbe8';ctx.beginPath();ctx.arc(xy.x,xy.y,2.6,0,Math.PI*2);ctx.fill();
      ctx.strokeStyle=color;ctx.lineWidth=1.5;ctx.beginPath();ctx.arc(xy.x,xy.y,m.seeker?6+2*Math.sin(performance.now()/120):5,0,Math.PI*2);ctx.stroke();
      ctx.fillStyle=color;ctx.font='600 10px ui-monospace,monospace';ctx.fillText('M'+m.uid+(m.seeker?' 主动':''),xy.x+9,xy.y-6);}
    if(sel&&sel.alive&&sel.mode==='pilot'&&P.state.status!=='setup')hud({...sel,position_m:disp(sel,dt)},w,h);
    const span=P.view==='2d'?P.map.span:P.camera.radius;$('scale-label').textContent='网格 '+spacing/1000+' km · 视距 '+fmt(span/1000)+' km';
    requestAnimationFrame(draw);
  }
  function fit(){P.camera={x:0,y:0,z:2500,yaw:.45,elevation:.5,radius:mapHalf()*2.4,orbit:0};P.map={x:0,y:0,span:mapHalf()*2.2};$('follow').checked=false;}
  function changeView(view){P.view=view;$('view3d').className=view==='3d'?'active':'';$('view2d').className=view==='2d'?'active':'';$('map-hint').textContent=view==='3d'?'拖动旋转 · Shift + 拖动平移 · 滚轮缩放':'拖动空白处平移 · 滚轮缩放 · 右键下达目的地';canvas.setAttribute('aria-label',view==='3d'?'三维空战战场，点击飞机选择，拖动旋转，滚轮缩放':'俯视战术地图，可布置和拖动飞机');}
  function locate(event){const r=canvas.getBoundingClientRect();return{x:event.clientX-r.left,y:event.clientY-r.top};}
  function hit(x,y){return P.screen.filter(p=>Math.hypot(p.x-x,p.y-y)<25).sort((a,b)=>Math.hypot(a.x-x,a.y-y)-Math.hypot(b.x-x,b.y-y))[0];}
  function deploy(team,world){
    if(P.scene.teams[team].length>=P.catalog.max_team){message('每队最多 '+P.catalog.max_team+' 架',true);return;}
    const p=clone(P.scene.teams[team].at(-1));p.position_m[0]=world[0];p.position_m[1]=world[1];P.scene.teams[team].push(p);P.dirty=true;P.pending=null;P.selected=team===0?P.scene.teams[0].length-1:P.scene.teams.flat().length-1;P.formSignature='';render();
  }
  async function destination(world){const p=selected();if(!p||!p.alive)return;await command('order',{plane:p.id,order:{mode:'manual',destination_m:world.slice(0,2),altitude_m:Number($('altitude').value),speed_mps:Number($('velocity').value)}});P.pending=null;P.formSignature='';message('已下达 #'+p.id+' 目的地：东 '+fmt(world[0]/1000,1)+' / 北 '+fmt(world[1]/1000,1)+' km');}
  canvas.addEventListener('pointerdown',e=>{
    if(e.button!==0)return;canvas.focus();const pos=locate(e),target=hit(pos.x,pos.y);
    if(P.pending){const world=worldAt(pos.x,pos.y,0);if(world){world[0]=clamp(world[0],-mapHalf(),mapHalf());world[1]=clamp(world[1],-mapHalf(),mapHalf());if(P.pending.type==='deploy')deploy(P.pending.team,world);else act(()=>destination(world));}return;}
    if(target)select(target.id);
    P.drag={...pos,startX:pos.x,startY:pos.y,unit:P.state.status==='setup'&&target?target.id:null,moved:false,pan:e.shiftKey};canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener('pointermove',e=>{
    const pos=locate(e),world=worldAt(pos.x,pos.y,0);if(world)$('coordinates').textContent='东 '+fmt(world[0]/1000,1)+' km · 北 '+fmt(world[1]/1000,1)+' km · ENU 米制';
    const d=P.drag;if(!d)return;
    const dx=pos.x-d.x,dy=pos.y-d.y;d.moved ||= Math.hypot(pos.x-d.startX,pos.y-d.startY)>4;
    if(d.unit!=null&&d.moved){const p=selectedDraft(),point=worldAt(pos.x,pos.y,p.position_m[2]);if(point){p.position_m[0]=clamp(point[0],-P.scene.map_half_m,P.scene.map_half_m);p.position_m[1]=clamp(point[1],-P.scene.map_half_m,P.scene.map_half_m);P.dirty=true;setField('east',fmt(p.position_m[0]/1000,2));setField('north',fmt(p.position_m[1]/1000,2));}}
    else if(P.view==='3d'&&!d.pan){const chase=$('follow').checked&&selected()?.mode==='pilot';if(chase)P.camera.orbit-=dx*.007;else P.camera.yaw-=dx*.007;P.camera.elevation=clamp(P.camera.elevation+dy*.006,-.35,1.48);}
    else {const scale=(P.view==='2d'?P.map.span:P.camera.radius)/Math.min(canvas.clientWidth,canvas.clientHeight);if(P.view==='2d'){P.map.x-=dx*scale;P.map.y+=dy*scale;}else{const b=cameraBasis();P.camera.x-=dx*scale*b.right[0];P.camera.y-=dx*scale*b.right[1];P.camera.x+=dy*scale*b.forward[0];P.camera.y+=dy*scale*b.forward[1];}$('follow').checked=false;}
    d.x=pos.x;d.y=pos.y;
  });
  canvas.addEventListener('pointerup',()=>{P.drag=null;render();});
  canvas.addEventListener('pointercancel',()=>{P.drag=null;});
  canvas.addEventListener('contextmenu',e=>{e.preventDefault();if(P.state.status==='setup'||P.state.status==='playback'||selected()?.tracked)return;const pos=locate(e),world=worldAt(pos.x,pos.y,0);if(world)act(()=>destination(world));});
  canvas.addEventListener('wheel',e=>{e.preventDefault();const factor=Math.exp(e.deltaY*.001);if(P.view==='3d')P.camera.radius=clamp(P.camera.radius*factor,600,400000);else P.map.span=clamp(P.map.span*factor,5000,450000);},{passive:false});
  $('view3d').onclick=()=>changeView('3d');$('view2d').onclick=()=>changeView('2d');$('fit').onclick=fit;
  $('add-blue').onclick=()=>{P.pending={type:'deploy',team:0};render();message('在地图点击蓝队飞机的部署位置');};
  $('add-orange').onclick=()=>{P.pending={type:'deploy',team:1};render();message('在地图点击橙队飞机的部署位置');};
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
  $('apply').onclick=()=>act(async()=>{if(P.state.status==='setup'){editDraft();await configure();message('初始状态已应用');}else{releaseKeys();await command('order',{plane:P.selected,order:{mode:'manual',destination_m:null,heading_deg:Number($('heading').value),altitude_m:Number($('altitude').value),speed_mps:Number($('velocity').value)}});P.formSignature='';message('导航命令已下达');}});
  $('destination').onclick=()=>{P.pending={type:'destination'};message('点击地图设置所选飞机的水平目的地；目标高度与速度使用右侧数值。按 Esc 取消。');};
  async function enterPilot(){releaseKeys();await command('speed',{speed:1});await command('order',{plane:P.selected,order:{mode:'pilot'}});P.formSignature='';$('follow').checked=true;P.camera.radius=1800;P.camera.elevation=.12;P.camera.orbit=0;const p=selected();P.camera.yaw=-(p.heading_deg||0)*Math.PI/180;changeView('3d');canvas.focus();message('已进入驾驶：点击战场后用 W/S 俯仰、A/D 滚转、R/F 或 ＋/－ 油门、空格减速板、按住 L 改平。拖动环视，C 回正视角。');}
  $('pilot').onclick=()=>act(async()=>{if(selected().mode==='pilot'){releaseKeys();await command('order',{plane:P.selected,order:{mode:'auto'}});P.formSignature='';}else await enterPilot();});
  $('mode').onchange=()=>{const mode=$('mode').value;act(async()=>{if(mode==='pilot')await enterPilot();else{releaseKeys();await command('order',{plane:P.selected,order:{mode}});P.formSignature='';}});};
  $('fire').onclick=()=>act(()=>command('fire',{plane:P.selected,track:Number($('tracks').value)}));
  $('drop-chaff').onclick=()=>act(()=>command('chaff',{plane:P.selected}));
  async function loadReplays(){try{P.replays=await api('/api/replays');}catch(error){message('无法读取回放列表：'+error.message,true);}render();}
  $('mode-replay').onclick=()=>{if(P.mode==='replay')return;if(P.state.status!=='setup'){message('请先重置当前推演，再切换到 WT 回放',true);return;}P.mode='replay';P.selected=-1;P.pending=null;loadReplays();render();};
  $('mode-setup').onclick=()=>{if(P.mode==='setup')return;releaseKeys();act(async()=>{if(P.state.replay_file||inReplay())await command('unload');P.mode='setup';P.selected=0;P.formSignature='';$('fork-panel').hidden=true;fit();});};
  $('me').onchange=()=>{P.me=$('me').value;store('wt-sandbox-me',P.me);render();};
  $('show-ai').onchange=()=>{P.showAI=$('show-ai').checked;render();};
  $('tl-range').addEventListener('input',()=>{P.seeking=true;$('tl-label').textContent=clockText(Number($('tl-range').value))+' / '+clockText(P.state.replay?.duration_s||0);});
  $('tl-range').addEventListener('change',()=>{const t=Number($('tl-range').value);act(async()=>{try{await command('seek',{t});}finally{P.seeking=false;}});});
  $('fork').onclick=openFork;$('fork-cancel').onclick=()=>{$('fork-panel').hidden=true;};
  $('fork-go').onclick=()=>act(async()=>{
    const s=P.state,control={};for(const sel of $('fork-rows').querySelectorAll('select'))if(sel.value!=='track')control[sel.dataset.id]=sel.value;
    const me=s.planes.find(p=>isMe(p)),includeAI=$('fork-ai').checked;$('fork-panel').hidden=true;
    message('正在建立推演世界（首次加载各机型飞行模型可能需要十几秒）…');
    const r=await command('fork',{t:s.time_s,control,include_ai:includeAI,me:me?me.id:null},2500);
    if(includeAI)P.showAI=true;
    if(r&&r.me!=null){P.selected=r.me;P.formSignature='';if(control[me.id]==='pilot'){$('follow').checked=true;P.camera.radius=1800;P.camera.elevation=.12;P.camera.orbit=0;P.camera.yaw=-(me.heading_deg||0)*Math.PI/180;changeView('3d');canvas.focus();}}
    message('接管推演：共 '+r.planes+' 架。'+(r.t0<r.t_fork-1e-6?'从 '+clockText(r.t0)+' 起重飞在途导弹，全速快进到 '+clockText(r.t_fork)+' 后自动暂停；点「继续推演」开始接管。':'已停在 '+clockText(r.t_fork)+'；点「继续推演」开始接管。'));});

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
        for(const a of groups.get(g).sort((x,y)=>title(x.id).localeCompare(title(y.id),'zh')))option(og,a.id,title(a.id)+(names[a.id]&&family(a.id)!=='中国'?'（'+names[a.id][1]+'）':'')+' — '+a.missiles.map(m=>missileName(m.id)).join(' / '));$('aircraft').append(og);}for(const a of P.catalog.archetypes)option($('archetype'),a,archetypes[a]||a);for(const s of P.catalog.skills)option($('skill'),s,s==='top'?'高手':'普通');await poll();if(P.mode==='replay')await loadReplays();$('preset').value=Object.keys(P.catalog.presets).find(k=>P.catalog.presets[k].name===P.scene.name)||'duel';$('seed').value=P.scene.seed;$('limit').value=P.scene.time_limit_s;fit();draw();
      while(true){await sleep(50);try{await poll();}catch(error){message('连接中断：'+error.message,true);await sleep(1200);}}
    }catch(error){message('启动失败：'+error.message,true);}
  }
  init();
})();
