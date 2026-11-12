from pathlib import Path
import shutil, py_compile
ROOT=Path(__file__).resolve().parent
INFO=ROOT/'information_gain.py'; MAIN=ROOT/'main.py'; CONFIG=ROOT/'pipeline_config.py'
for p in (INFO,MAIN,CONFIG):
    if not p.exists(): raise FileNotFoundError(p)
    b=p.with_suffix(p.suffix+'.bak')
    if not b.exists(): shutil.copy2(p,b)

def rep(s,old,new,label):
    if new in s: return s
    if old not in s: raise RuntimeError(f'Could not patch {label}')
    return s.replace(old,new,1)

cfg=CONFIG.read_text(encoding='utf-8')
if 'PRECIP_WET_THRESHOLD_MM' not in cfg:
    cfg += '\n# Daily precipitation threshold for wet/dry classification.\nPRECIP_WET_THRESHOLD_MM = 0.1\n'
CONFIG.write_text(cfg,encoding='utf-8')

s=INFO.read_text(encoding='utf-8')
s=rep(s,'from sklearn.linear_model import LinearRegression','from sklearn.linear_model import LinearRegression, LogisticRegression','import')
if 'brier_score_loss' not in s:
    s=s.replace('from sklearn.metrics import ','from sklearn.metrics import brier_score_loss, ',1)
helper = '''

def directional_precip_occurrence_error(train, valid, source_col, target_col):
    # Predict wet probability from the continuous source precipitation amount.
    y_train=(train.iloc[:,target_col].to_numpy(float)>=PRECIP_WET_THRESHOLD_MM).astype(np.int8)
    y_valid=(valid.iloc[:,target_col].to_numpy(float)>=PRECIP_WET_THRESHOLD_MM).astype(np.int8)
    x_train=np.log1p(train.iloc[:,[source_col]].clip(lower=0).to_numpy(float))
    x_valid=np.log1p(valid.iloc[:,[source_col]].clip(lower=0).to_numpy(float))
    if np.unique(y_train).size < 2:
        probability_wet=np.full(len(valid),y_train.mean(),dtype=float)
    else:
        classifier=LogisticRegression(max_iter=1000,class_weight="balanced",random_state=RANDOM_STATE)
        classifier.fit(x_train,y_train)
        probability_wet=classifier.predict_proba(x_valid)[:,1]
    return brier_score_loss(y_valid,np.clip(probability_wet,0.0,1.0))
'''
if 'def directional_precip_occurrence_error' not in s:
    s=rep(s,'\ndef build_pairs(',helper+'\ndef build_pairs(','helper')
branch = '''
        elif metric == "precipitation_occurrence_error":
            if name != "precipitation":
                raise ValueError("precipitation_occurrence_error requires --phenomenon precipitation")
            value_1_to_2=directional_precip_occurrence_error(train,valid,0,1)
            value_2_to_1=directional_precip_occurrence_error(train,valid,1,0)
'''
if 'elif metric == "precipitation_occurrence_error"' not in s:
    needle='        target_values.append((value_1_to_2 + value_2_to_1) / 2)'
    s=rep(s,needle,branch+'\n'+needle,'pair branch')
if '"precip_wet_threshold_mm"' not in s:
    needle='"years":years,"phenomenon":name}'
    repl='"years":years,"phenomenon":name,\n              "precip_wet_threshold_mm": PRECIP_WET_THRESHOLD_MM if name == "precipitation" else None}'
    s=rep(s,needle,repl,'metadata')
INFO.write_text(s,encoding='utf-8')

m=MAIN.read_text(encoding='utf-8')
if '"precipitation_occurrence_error"' not in m:
    m=rep(m,'"temperature_class_error"','"temperature_class_error",\n        "precipitation_occurrence_error"','CLI metric')
MAIN.write_text(m,encoding='utf-8')
for p in (INFO,MAIN,CONFIG): py_compile.compile(str(p),doraise=True)
print('Wet/dry metric enabled. Backups were created.')
print('Run: python main.py --phenomenon precipitation --metric precipitation_occurrence_error --years 1 --from-step information --skip-download --force --test-candidates 50 --test-targets 500')
