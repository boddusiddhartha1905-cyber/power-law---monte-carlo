#!/usr/bin/env python3
"""Rolling-origin four-model backtest for Titan and Jio Financial.
Uses expanding windows and globally unique evaluation dates to avoid fold-index collisions.
"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
import matplotlib.pyplot as plt

PROJECT_DIR = Path.home() / "Documents" / "power law - monte carlo"
DATA_DIR = PROJECT_DIR / "asset_data"
ASSETS = {
    "titan": ("titan_train.csv", "titan_test.csv"),
    "jio_financial": ("jio_financial_train.csv", "jio_financial_test.csv"),
}
INITIAL_TRAIN_FRACTION = 0.60
N_PATHS_CRPS = 2500
SEED = 20261023
RAW_TAIL_THRESHOLD = 0.025
STD_TAIL_THRESHOLD = 2.0

def find_column(df, candidates):
    cols={str(c).strip().lower().replace(" ","_"):c for c in df.columns}
    for candidate in candidates:
        key=candidate.strip().lower().replace(" ","_")
        if key in cols:return cols[key]
    for c in df.columns:
        name=str(c).strip().lower().replace(" ","_")
        if any(x.strip().lower().replace(" ","_") in name for x in candidates):return c
    return None

def load_returns(path):
    if not path.exists(): raise FileNotFoundError(f"Missing {path}")
    df=pd.read_csv(path)
    dc=find_column(df,["date","datetime","timestamp"])
    rc=find_column(df,["log_return","log_returns","return","returns","daily_return"])
    if rc is None: raise ValueError(f"No return column in {path.name}")
    r=pd.to_numeric(df[rc],errors="coerce")
    dates=pd.to_datetime(df[dc],errors="coerce") if dc is not None else pd.Series(pd.NaT,index=df.index)
    out=pd.DataFrame({"date":dates,"return":r}).dropna(subset=["return"]).reset_index(drop=True)
    if out["date"].notna().all():
        out=out.sort_values("date").reset_index(drop=True)
    return out

def fit_hybrid(r, threshold):
    r=np.asarray(r,float)
    r=r[np.isfinite(r)]
    abs_r=np.abs(r)
    mask=abs_r>=threshold
    # Small expanding training folds may not contain 30 observations beyond
    # the requested cutoff. Lower the cutoff using training data only until
    # there are at least 30 tail observations; never use the forecast target.
    min_tail=30
    used_threshold=float(threshold)
    if int(mask.sum())<min_tail:
        if len(r)<=min_tail:
            raise ValueError(f"Need more than {min_tail} finite training observations; found {len(r)}")
        ordered=np.sort(abs_r)
        used_threshold=min(float(threshold), float(ordered[-min_tail]))
        mask=abs_r>=used_threshold
        # Handle tied values at the selected cutoff.
        while int(mask.sum())<min_tail:
            used_threshold=float(np.nextafter(used_threshold, -np.inf))
            mask=abs_r>=used_threshold
    tail=abs_r[mask]
    denom=np.sum(np.log(tail/used_threshold))
    if not np.isfinite(denom) or denom<=0:
        raise ValueError("Cannot estimate Pareto tail: insufficient variation in tail observations")
    alpha=len(tail)/denom
    return {"threshold":used_threshold,"requested_threshold":float(threshold),
            "threshold_adjusted":bool(used_threshold < threshold),
            "alpha":float(alpha),"p_tail":float(mask.mean()),
            "neg":float(np.mean(r[mask]<0)),"body":r[~mask],"tail_n":int(mask.sum())}

def sample_hybrid(params,n,rng):
    tail=rng.random(n)<params["p_tail"]; x=np.empty(n)
    nb=int((~tail).sum()); nt=int(tail.sum())
    if nb:x[~tail]=rng.choice(params["body"],size=nb,replace=True)
    if nt:
        u=np.clip(rng.random(nt),0,1-1e-12)
        mag=params["threshold"]*(1-u)**(-1/params["alpha"])
        x[tail]=np.where(rng.random(nt)<params["neg"],-mag,mag)
    return x

def hybrid_logpdf(x,params):
    x=np.asarray(x,float); out=np.empty_like(x); tail=np.abs(x)>=params["threshold"]
    if np.any(tail):
        signprob=np.where(x[tail]<0,params["neg"],1-params["neg"])
        den=params["p_tail"]*signprob*params["alpha"]*params["threshold"]**params["alpha"]
        den=den/np.maximum(np.abs(x[tail]),params["threshold"])**(params["alpha"]+1)
        out[tail]=np.log(np.maximum(den,1e-300))
    if np.any(~tail):
        body=params["body"]; sd=max(float(np.std(body,ddof=1)),1e-6)
        bw=max(1.06*sd*max(len(body),2)**(-.2),1e-5)
        dif=(x[~tail,None]-body[None,:])/bw
        den=np.exp(-.5*dif**2).mean(axis=1)/(bw*np.sqrt(2*np.pi))
        den=(1-params["p_tail"])*den
        out[~tail]=np.log(np.maximum(den,1e-300))
    return out

def fit_garch(r):
    y=np.asarray(r,float)*100; mu0=float(y.mean()); v0=max(float(np.var(y-mu0)),1e-6)
    def unpack(theta):
        omega=np.exp(np.clip(theta[0],-20,10))
        alpha=1/(1+np.exp(-np.clip(theta[1],-20,20)))
        br=1/(1+np.exp(-np.clip(theta[2],-20,20)))
        beta=br*max(1e-8,.999-alpha)
        return float(theta[3]),float(omega),float(alpha),float(beta)
    def nll(theta):
        mu,omega,alpha,beta=unpack(theta); eps=y-mu; h=np.empty_like(y)
        h[0]=max(v0,omega/max(1e-5,1-alpha-beta))
        for i in range(1,len(y)):h[i]=omega+alpha*eps[i-1]**2+beta*h[i-1]
        h=np.maximum(h,1e-10)
        return float(.5*np.sum(np.log(2*np.pi)+np.log(h)+eps**2/h))
    init=np.array([np.log(max(v0*.05,1e-5)),-1.5,2.0,mu0])
    opt=minimize(nll,init,method="Nelder-Mead",options={"maxiter":1500,"xatol":1e-5,"fatol":1e-5})
    if not opt.success:warnings.warn(f"GARCH optimizer: {opt.message}")
    mu,omega,alpha,beta=unpack(opt.x); eps=y-mu; h=np.empty_like(y)
    h[0]=max(v0,omega/max(1e-5,1-alpha-beta))
    for i in range(1,len(y)):h[i]=omega+alpha*eps[i-1]**2+beta*h[i-1]
    z=eps/np.sqrt(np.maximum(h,1e-10)); z=(z-z.mean())/max(float(z.std(ddof=1)),1e-8)
    return {"mu":mu,"omega":omega,"alpha":alpha,"beta":beta,"eps_last":float(eps[-1]),
            "var_last":float(h[-1]),"z":z}

def crps_ensemble(samples,obs):
    x=np.asarray(samples,float); first=np.mean(np.abs(x-obs)); xs=np.sort(x); n=len(xs)
    idx=np.arange(1,n+1); half_pair=np.sum((2*idx-n-1)*xs)/(n*n)
    return float(first-half_pair)

def simulate_garch_one_step(g,n,rng,hybrid=None):
    var_next=max(float(g["omega"]+g["alpha"]*g["eps_last"]**2+g["beta"]*g["var_last"]),1e-10)
    if hybrid is None:z=rng.standard_normal(n)
    else:
        z=sample_hybrid(hybrid,n,rng)
        z=(z-hybrid["norm_mean"])/hybrid["norm_sd"]
    return (g["mu"]+np.sqrt(var_next)*z)/100.0

def score_window(train, observations, dates, rng, asset, period, fold, train_n):
    rows=[]; prev=np.asarray(train,float).copy(); cache=None
    for j,obs in enumerate(np.asarray(observations,float)):
        if j%60==0 or cache is None:
            mu=float(prev.mean()); sd=max(float(prev.std(ddof=1)),1e-6)
            raw=fit_hybrid(prev,RAW_TAIL_THRESHOLD); g=fit_garch(prev)
            zh=fit_hybrid(g["z"],STD_TAIL_THRESHOLD)
            cal=sample_hybrid(zh,100_000,rng)
            zh["norm_mean"]=float(cal.mean()); zh["norm_sd"]=max(float(cal.std(ddof=1)),1e-6)
            cache=(mu,sd,raw,g,zh)
        mu,sd,raw,g,zh=cache
        ns=rng.normal(mu,sd,N_PATHS_CRPS); hs=sample_hybrid(raw,N_PATHS_CRPS,rng)
        gn=simulate_garch_one_step(g,N_PATHS_CRPS,rng)
        gh=simulate_garch_one_step(g,N_PATHS_CRPS,rng,zh)
        sigma_pct=np.sqrt(max(g["omega"]+g["alpha"]*g["eps_last"]**2+g["beta"]*g["var_last"],1e-10))
        zobs=(obs*100-g["mu"])/sigma_pct
        scores={
            "Normal IID":(-float(norm.logpdf(obs,loc=mu,scale=sd)),crps_ensemble(ns,obs),ns),
            "Power-law hybrid IID":(-float(hybrid_logpdf(np.array([obs]),raw)[0]),crps_ensemble(hs,obs),hs),
            "GARCH-normal":(-float(norm.logpdf(obs,loc=g["mu"]/100,scale=sigma_pct/100)),crps_ensemble(gn,obs),gn),
            # Density conversion: f_return = f_z * 100/sigma_pct; NLL = -log(f_z) - log(100/sigma_pct)
            "GARCH-power-law hybrid":(-float(hybrid_logpdf(np.array([zobs]),zh)[0])-np.log(100/sigma_pct),crps_ensemble(gh,obs),gh)
        }
        date = pd.Timestamp(dates.iloc[j]) if dates is not None and pd.notna(dates.iloc[j]) else pd.NaT
        for name,(nll,crps,draws) in scores.items():
            q05=float(np.quantile(draws,.05))
            rows.append({"asset":asset,"evaluation_period":period,"fold":fold,"train_n":train_n,
                         "date":date,"day_index":j,"Model":name,"actual_return":obs,
                         "negative_log_score":nll,"CRPS":crps,"VaR95_loss":-q05,
                         "VaR95_exceedance":int(obs<q05)})
        prev=np.append(prev,obs)
        # Update variance state sequentially after observing today's return.
        eps_now=obs*100-g["mu"]
        var_now=g["omega"]+g["alpha"]*g["eps_last"]**2+g["beta"]*g["var_last"]
        g["eps_last"]=float(eps_now); g["var_last"]=float(max(var_now,1e-10))
    return pd.DataFrame(rows)

def run_asset(asset, files):
    out_dir=PROJECT_DIR/f"{asset}_rolling_backtest_results"; out_dir.mkdir(parents=True,exist_ok=True)
    train_df=load_returns(DATA_DIR/files[0]); test_df=load_returns(DATA_DIR/files[1])
    train=train_df["return"].to_numpy(float); test=test_df["return"].to_numpy(float)
    train_dates=train_df["date"]; test_dates=test_df["date"]
    if len(train)<250 or len(test)<50: raise ValueError(f"{asset}: not enough data")
    initial=int(len(train)*INITIAL_TRAIN_FRACTION); remaining=len(train)-initial
    block=max(40,remaining//4); rng=np.random.default_rng(SEED)
    frames=[]; start=initial; fold=1
    while start<len(train):
        end=min(start+block,len(train))
        if end-start<40:break
        frames.append(score_window(train[:start],train[start:end],train_dates.iloc[start:end].reset_index(drop=True),
                                   rng,asset,"within_training",fold,start))
        start=end; fold+=1
    frames.append(score_window(train,test,test_dates.reset_index(drop=True),rng,asset,
                               "existing_test_exploratory",0,len(train)))
    scores=pd.concat(frames,ignore_index=True)
    scores.to_csv(out_dir/"daily_scores.csv",index=False)
    summary=(scores.groupby(["asset","evaluation_period","Model"],as_index=False)
        .agg(days=("actual_return","size"),mean_negative_log_score=("negative_log_score","mean"),
             mean_CRPS=("CRPS","mean"),VaR95_exceedances=("VaR95_exceedance","sum"),
             VaR95_expected_exceedances=("VaR95_exceedance",lambda x:len(x)*.05),
             VaR95_exceedance_rate=("VaR95_exceedance","mean"),mean_VaR95_loss=("VaR95_loss","mean")))
    summary.to_csv(out_dir/"score_summary.csv",index=False)
    fold_summary=(scores.groupby(["asset","evaluation_period","fold","Model"],as_index=False)
        .agg(days=("actual_return","size"),mean_negative_log_score=("negative_log_score","mean"),
             mean_CRPS=("CRPS","mean"),VaR95_exceedance_rate=("VaR95_exceedance","mean")))
    fold_summary.to_csv(out_dir/"fold_summary.csv",index=False)
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    p=summary[summary.evaluation_period=="within_training"]
    if len(p):
        axes[0].bar(p["Model"],p["mean_CRPS"]);axes[0].set_title(f"{asset}: rolling validation CRPS")
        axes[0].tick_params(axis="x",rotation=25);axes[0].set_ylabel("CRPS (lower is better)")
    p=summary[summary.evaluation_period=="existing_test_exploratory"]
    if len(p):
        axes[1].bar(p["Model"],p["mean_negative_log_score"]);axes[1].set_title(f"{asset}: exploratory test NLL")
        axes[1].tick_params(axis="x",rotation=25);axes[1].set_ylabel("NLL (lower is better)")
    fig.tight_layout();fig.savefig(out_dir/"score_comparison.png",dpi=160);plt.close(fig)
    print(f"\n{asset.upper()} ROLLING BACKTEST")
    print(summary.to_string(index=False,float_format=lambda x:f"{x:.6f}"))
    print("Unique within-training dates:",scores.loc[scores.evaluation_period=="within_training","date"].nunique())
    print("Saved:",out_dir)

def main():
    for asset,files in ASSETS.items():run_asset(asset,files)
    print("\nBoth rolling backtests finished. Dates are recorded explicitly to prevent fold index collisions.")
    print("The existing test period is exploratory because it has already been examined.")

if __name__=="__main__":main()
