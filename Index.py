# Imports
import re, math, random, itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split

import nltk
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.util import ngrams
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import Laplace

import spacy
import textstat
from scipy import stats

RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
nlp.add_pipe("sentencizer")
STOPWORDS = set(stopwords.words("english"))
sia = SentimentIntensityAnalyzer()

sns.set_theme(style="whitegrid", palette="Set2")
plt.rcParams["figure.dpi"] = 110
plot_number = 0



FIGURE_DIR = "figures"
os.makedirs(FIGURE_DIR, exist_ok=True)

plot_number = 0

def show_plot():
    global plot_number
    plot_number += 1

    fig = plt.gcf()

    filename = os.path.join(
        FIGURE_DIR,
        f"figure_{plot_number}.png"
    )

    
    fig.savefig(filename, dpi=300, bbox_inches="tight")

    print(f"Saved: {filename}", flush=True)

    
    plt.show(block=False)
    plt.pause(2)


    plt.close(fig)


print("Environment ready.", flush=True)

# Loading RAID document. 
N_PER_CELL = 60
DOMAINS = ["reviews", "news", "recipes"]
RANDOM_STATE = 42


def sample_real_raid(chunks, n_per_cell=N_PER_CELL, seed=RANDOM_STATE):
    
    rng = random.Random(seed)
    cells = {(domain, label): [] for domain in DOMAINS for label in ("human", "ai")}
    seen = dict.fromkeys(cells, 0)
    required = ["id", "domain", "model", "generation"]
    scanned = 0
    for chunk in chunks:
        missing = set(required) - set(chunk.columns)
        if missing:
            raise ValueError(
                f"RAID input lacks required columns: {sorted(missing)}. "
                "The public test split is unlabeled; use the labeled training split."
            )
        scanned += len(chunk)
        eligible = chunk.loc[chunk["domain"].isin(DOMAINS), required].dropna()
        eligible = eligible.loc[
            eligible["generation"].str.strip().ne("")
            & eligible["model"].str.strip().ne("")
        ]
        for doc_id, domain, model, generation in eligible.itertuples(index=False, name=None):
            label = "human" if model.strip().lower() == "human" else "ai"
            key = (domain, label)
            seen[key] += 1
            row = dict(id=doc_id, domain=domain, model=model,
                       label=label, text=generation)
            if len(cells[key]) < n_per_cell:
                cells[key].append(row)
            else:
                j = rng.randrange(seen[key])
                if j < n_per_cell:
                    cells[key][j] = row
        if scanned % 100000 == 0:
            print(f"Scanned {scanned:,} real rows; retained "
                  f"{sum(map(len, cells.values()))} documents", flush=True)
    too_small = {f"{d}/{label}": len(rows) for (d, label), rows in cells.items()
                 if len(rows) < n_per_cell}
    if too_small:
        raise ValueError(f"Need {n_per_cell} valid documents per domain/class; "
                         f"insufficient cells: {too_small}")
    print(f"Finished scanning {scanned:,} real rows.", flush=True)
    return pd.DataFrame([row for rows in cells.values() for row in rows])


def try_load_real_raid(source):

    import urllib.request
    print(f"Loading  RAID data from {source}", flush=True)
    
    stream = (urllib.request.urlopen(source, timeout=120)
              if source.startswith(("https://", "http://"))
              else open(source, "rb"))
    with stream:
        with pd.read_csv(stream, chunksize=10000, dtype=str, keep_default_na=False) as chunks:
            return sample_real_raid(chunks)



source = "https://dataset.raid-bench.xyz/train_none.csv"

df_real = try_load_real_raid(source)
df = df_real.copy()
DATA_SOURCE = f"real RAID labeled training sample: {source}"
print(f"Data source: {DATA_SOURCE}")
print(f"Stratified subsample shape: {df.shape}")
print(df.head())


# Class balance, domain balance, missing values, document-length distribution
print("Missing values per column:\n", df.isna().sum(), "\n")
print("Class balance:\n", df["label"].value_counts(), "\n")
print("Domain balance:\n", df["domain"].value_counts(), "\n")
print("Class x Domain crosstab:\n", pd.crosstab(df["domain"], df["label"]), "\n")

df["word_count"] = df["text"].str.split().apply(len)
df["char_count"] = df["text"].str.len()

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
sns.histplot(data=df, x="word_count", hue="label", kde=True, ax=axes[0], element="step")
axes[0].set_title("Document length (words) by class")
sns.boxplot(data=df, x="domain", y="word_count", hue="label", ax=axes[1])
axes[1].set_title("Document length by domain and class")
plt.tight_layout()
show_plot()

print("\nDocument length statistics:")
print(df[["word_count", "char_count"]].describe().to_string())



def light_clean(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)          # strip HTML fragments
    text = re.sub(r"\s+", " ", text).strip()       # collapse whitespace
    return text

def heavy_clean(text: str) -> str:
    text = light_clean(text).lower()
    doc = nlp(text)
    tokens = [tok.lemma_ for tok in doc
              if tok.is_alpha and tok.lemma_ not in STOPWORDS and len(tok.lemma_) > 1]
    return " ".join(tokens)

df["light_text"] = df["text"].apply(light_clean)
df = df[df["light_text"].str.len() > 0].reset_index(drop=True)
df["clean_text"] = df["light_text"].apply(heavy_clean)
df = df[df["clean_text"].str.split().apply(len) > 2].reset_index(drop=True)
if len(df) < 6 or set(df["label"]) != {"human", "ai"}:
    raise ValueError("Too few valid documents or a missing class after cleaning.")
print("Class/domain counts after cleaning:\n", pd.crosstab(df["domain"], df["label"]))

print(df[["text", "light_text", "clean_text"]].iloc[0].to_dict())
print(f"\nFinal cleaned corpus size: {len(df)} documents")



def readability_scores(text: str) -> pd.Series:
    return pd.Series({
        "flesch_reading_ease": textstat.flesch_reading_ease(text),
        "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),
        "gunning_fog": textstat.gunning_fog(text),
        "avg_sentence_length": textstat.avg_sentence_length(text),
    })

read_df = df["light_text"].apply(readability_scores)
df = pd.concat([df, read_df], axis=1)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, col in zip(axes, ["flesch_reading_ease", "flesch_kincaid_grade", "gunning_fog"]):
    sns.boxplot(data=df, x="label", y=col, ax=ax)
    ax.set_title(col)
plt.tight_layout()
show_plot()



print("\nReadability :- ")
print( df.groupby("label")[["flesch_reading_ease","flesch_kincaid_grade","gunning_fog"]].agg(["mean", "std"]).to_string())

def sentiment_scores(text: str) -> pd.Series:
    vs = sia.polarity_scores(text)
    return pd.Series({
        "sentiment_polarity": vs["compound"],
        "sentiment_pos": vs["pos"],
        "sentiment_neg": vs["neg"],
        "sentiment_neu": vs["neu"],
    })

sent_df = df["light_text"].apply(sentiment_scores)
df = pd.concat([df, sent_df], axis=1)

fig, ax = plt.subplots(figsize=(6, 4))
sns.violinplot(data=df, x="label", y="sentiment_polarity", ax=ax, cut=0)
ax.set_title("Sentiment polarity (VADER compound) by class")
plt.tight_layout()
show_plot()


print("\nSentiment polarity")
print(df.groupby("label")["sentiment_polarity"].agg(["mean", "std", "median"]).to_string())


tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 1), min_df=2)
X_tfidf = tfidf.fit_transform(df["clean_text"])
feature_names = np.array(tfidf.get_feature_names_out())

def top_terms(label, k=15):
    mask = (df["label"] == label).values
    mean_scores = np.asarray(X_tfidf[mask].mean(axis=0)).ravel()
    top_idx = mean_scores.argsort()[::-1][:k]
    return pd.DataFrame({"term": feature_names[top_idx], "mean_tfidf": mean_scores[top_idx]})

human_terms, ai_terms = top_terms("human"), top_terms("ai")
print("Top human terms:\n", human_terms.to_string(index=False))
print("\nTop AI terms:\n", ai_terms.to_string(index=False))

# n-grams (bigrams) distinguishing the two classes
cv = CountVectorizer(ngram_range=(2, 2), max_features=2000, min_df=2)
X_ngrams = cv.fit_transform(df["clean_text"])
ngram_names = np.array(cv.get_feature_names_out())
for label in ["human", "ai"]:
    mask = (df["label"] == label).values
    freqs = np.asarray(X_ngrams[mask].sum(axis=0)).ravel()
    top_idx = freqs.argsort()[::-1][:10]
    print(f"\nTop bigrams ({label}):", list(ngram_names[top_idx]))



def tokenize(text):
    return [t for t in re.findall(r"[a-z']+", text.lower())]

all_tokens = [tokenize(t) for t in df["light_text"]]
train_data, vocab = padded_everygram_pipeline(2, all_tokens)
lm = Laplace(2)
lm.fit(train_data, vocab)
# Perplexity
def doc_perplexity(tokens):
    if len(tokens) < 2:
        return np.nan
    test_ngrams = list(ngrams(tokens, 2, pad_left=True, pad_right=True,
                               left_pad_symbol="<s>", right_pad_symbol="</s>"))
    try:
        return lm.perplexity(test_ngrams)
    except Exception:
        return np.nan

df["perplexity"] = [doc_perplexity(toks) for toks in all_tokens]
df["perplexity"] = df["perplexity"].replace([np.inf, -np.inf], np.nan)

fig, ax = plt.subplots(figsize=(6, 4))
sns.boxplot(data=df, x="label", y="perplexity", ax=ax, showfliers=False)
ax.set_title("Bigram-LM perplexity by class")
plt.tight_layout()
show_plot()

print("\nPerplexity:")
print(df.groupby("label")["perplexity"].agg(["mean", "median", "std"]).to_string())

#Burstiness
def burstiness_metrics(row):
    sentences = re.split(r"(?<=[.!?])\s+", row["light_text"])
    sentences = [s for s in sentences if s.strip()]
    sent_lengths = [len(s.split()) for s in sentences]
    sent_len_var = np.var(sent_lengths) if len(sent_lengths) > 1 else 0.0

    content_tokens = [tok.lemma_.lower() for tok in nlp(row["light_text"])
                       if tok.is_alpha and tok.lemma_.lower() not in STOPWORDS]
    positions = {}
    for idx, tok in enumerate(content_tokens):
        positions.setdefault(tok, []).append(idx)
    gaps = []
    for idx_list in positions.values():
        if len(idx_list) > 1:
            gaps.extend(np.diff(idx_list))
    gap_var = np.var(gaps) if len(gaps) > 1 else 0.0

    return pd.Series({"sentence_len_var": sent_len_var, "repeat_gap_var": gap_var})

burst_df = df.apply(burstiness_metrics, axis=1)
df = pd.concat([df, burst_df], axis=1)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
sns.boxplot(data=df, x="label", y="sentence_len_var", ax=axes[0], showfliers=False)
axes[0].set_title("Sentence-length variance (burstiness) by class")
sns.boxplot(data=df, x="label", y="repeat_gap_var", ax=axes[1], showfliers=False)
axes[1].set_title("Repeated-content-word gap variance by class")
plt.tight_layout()
show_plot()

print("\nBurstiness statistics by class:")
print(df.groupby("label")[["sentence_len_var","repeat_gap_var"]].agg(["mean", "std"]).to_string())



# TruncatedSVD is the standard PCA analogue for sparse TF-IDF matrices (avoids densifying a large sparse matrix)
svd = TruncatedSVD(n_components=2, random_state=RANDOM_STATE)
pca_coords = svd.fit_transform(X_tfidf)
df["pca_1"], df["pca_2"] = pca_coords[:, 0], pca_coords[:, 1]

fig, ax = plt.subplots(figsize=(6.5, 5))
sns.scatterplot(data=df, x="pca_1", y="pca_2", hue="label", style="domain", alpha=0.7, ax=ax)
ax.set_title(f"TruncatedSVD of TF-IDF vectors (explained variance: {svd.explained_variance_ratio_.sum():.2%})")
plt.tight_layout()
show_plot()



# Reduce to 50 dims first (standard practice) to speed/stabilise t-SNE, then project non-linearly to 2D
svd50 = TruncatedSVD(n_components=min(50, X_tfidf.shape[1] - 1), random_state=RANDOM_STATE)
X_reduced = svd50.fit_transform(X_tfidf)

tsne = TSNE(n_components=2, perplexity=min(30, max(2, len(df) // 10), len(df) - 1),
            random_state=RANDOM_STATE, init="pca", learning_rate="auto")
tsne_coords = tsne.fit_transform(X_reduced)
df["tsne_1"], df["tsne_2"] = tsne_coords[:, 0], tsne_coords[:, 1]

fig, ax = plt.subplots(figsize=(6.5, 5))
sns.scatterplot(data=df, x="tsne_1", y="tsne_2", hue="label", style="domain", alpha=0.7, ax=ax)
ax.set_title("t-SNE of TF-IDF vectors (via 50-component SVD)")
plt.tight_layout()
show_plot()



FEATURES = ["word_count", "flesch_reading_ease", "flesch_kincaid_grade", "gunning_fog",
            "sentiment_polarity", "perplexity", "sentence_len_var", "repeat_gap_var"]

def cohens_d(a, b):
    na, nb = len(a), len(b)
    pooled_std = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return (a.mean() - b.mean()) / pooled_std if pooled_std > 0 else np.nan

def rank_biserial(u_stat, na, nb):
    return 1 - (2 * u_stat) / (na * nb)

results = []
for feat in FEATURES:
    a = df.loc[df["label"] == "human", feat].dropna()
    b = df.loc[df["label"] == "ai", feat].dropna()
    if len(a) < 3 or len(b) < 3:
        continue
    normal_a = stats.shapiro(a.sample(min(len(a), 500), random_state=RANDOM_STATE)).pvalue > 0.05
    normal_b = stats.shapiro(b.sample(min(len(b), 500), random_state=RANDOM_STATE)).pvalue > 0.05

    if normal_a and normal_b:
        stat, p = stats.ttest_ind(a, b, equal_var=False)
        effect = cohens_d(a, b)
        test_used = "Welch t-test"
    else:
        stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        effect = rank_biserial(stat, len(a), len(b))
        test_used = "Mann-Whitney U"

    results.append({"feature": feat, "test": test_used, "statistic": stat,
                     "p_value": p, "effect_size": effect,
                     "human_mean": a.mean(), "ai_mean": b.mean()})

results_df = pd.DataFrame(results)
reject, p_corrected, _, _ = __import__("statsmodels.stats.multitest", fromlist=["multipletests"]).multipletests(
    results_df["p_value"], method="fdr_bh"
) if False else (None, None, None, None)

# Benjamini-Hochberg without requiring statsmodels
def bh_correction(pvals):
    pvals = np.asarray(pvals)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    corrected = np.empty(n)
    corrected[order] = np.clip(ranked, 0, 1)
    return corrected

results_df["p_fdr_bh"] = bh_correction(results_df["p_value"].values)
results_df["significant_(fdr<0.05)"] = results_df["p_fdr_bh"] < 0.05
results_df = results_df.sort_values("p_fdr_bh")
print("\nStatistical comparisons:\n", results_df.to_string(index=False))

print(f"Completed analysis of {len(df)}  RAID documents; {plot_number} figures.", flush=True)

