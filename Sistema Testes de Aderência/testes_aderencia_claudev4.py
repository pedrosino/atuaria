"""
Aderência de Tábuas de Mortalidade
===================================
Testes: Qui-quadrado (agrupamento dinâmico), Kolmogorov-Smirnov, Z bilateral.
"""

import io
import warnings

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats
from scipy.stats import chi2
matplotlib.use("Agg")

warnings.filterwarnings("ignore")

try:
    from reportlab.lib import colors as rl_colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
    )
    REPORTLAB_OK = True
except ImportError:
    REPORTLAB_OK = False


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG & CSS
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Aderência de Tábuas de Mortalidade",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'IBM Plex Sans', Verdana, Geneva, sans-serif;
    font-size: 14px;
}

h1, h2, h3, h4 {
    font-family: 'IBM Plex Sans', Verdana, Geneva, sans-serif !important;
    font-weight: 600 !important;
}

.main-title {
    font-size: 1.55rem;
    font-weight: 700;
    color: #0f172a;
    letter-spacing: -0.01em;
    line-height: 1.2;
    margin-bottom: 0.15rem;
}
.sub-title {
    font-size: 0.85rem;
    color: #64748b;
    font-weight: 400;
    margin-bottom: 1.5rem;
}
.stat-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.8rem;
}
.stat-card .label {
    font-size: 0.67rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: #94a3b8;
    margin-bottom: 0.2rem;
}
.stat-card .value {
    font-size: 1.25rem;
    font-weight: 700;
    color: #0f172a;
}
.section-header {
    font-size: 0.95rem;
    font-weight: 700;
    color: #0f172a;
    border-left: 3px solid #3b82f6;
    padding-left: 0.65rem;
    margin: 1.2rem 0 0.8rem 0;
}
.info-box {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    color: #1e40af;
    font-size: 0.82rem;
    margin: 0.8rem 0;
}
div[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }

/* Sidebar compacta */
[data-testid="stSidebar"] { font-size: 13px; }
[data-testid="stSidebar"] .stSelectSlider label,
[data-testid="stSidebar"] .stNumberInput label { font-size: 0.78rem !important; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  CONSTANTES
# ═══════════════════════════════════════════════════════════════════════════════

SEXOS = ("M", "F")
LABEL_SEXO = {"M": "Masculino", "F": "Feminino"}


# ═══════════════════════════════════════════════════════════════════════════════
#  CARREGAMENTO — NOVO FORMATO WIDE
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def carregar_populacao(conteudo: bytes, nome: str) -> pd.DataFrame | None:
    """
    Formato wide:
      matricula | plano | idade | expostos_M | ocorridos_M | expostos_F | ocorridos_F
    Retorna formato long: matricula, plano, sexo, idade, expostos, ocorridos
    """
    buf = io.BytesIO(conteudo)
    df = pd.read_excel(buf) if nome.endswith(("xlsx", "xls")) else pd.read_csv(buf)
    df.columns = [str(c).strip() for c in df.columns]

    # Detecta se já está em formato long (tem coluna 'sexo')
    cols_low = [c.lower() for c in df.columns]
    if any("sexo" in c or c in ("sex","m","f","masc","fem") for c in cols_low):
        # Formato long legado — renomeia e normaliza
        rename = {}
        for c in df.columns:
            cl = c.lower()
            if "matr" in cl:   rename[c] = "matricula"
            elif "plan" in cl: rename[c] = "plano"
            elif "sex"  in cl: rename[c] = "sexo"
            elif "idade" in cl or cl == "id": rename[c] = "idade"
            elif "exp"  in cl: rename[c] = "expostos"
            elif any(p in cl for p in ("ocor","obit","mort")): rename[c] = "ocorridos"
        df.rename(columns=rename, inplace=True)
        df["sexo"] = df["sexo"].apply(_normalizar_sexo)
    else:
        # Formato wide novo
        rename = {}
        for c in df.columns:
            cl = c.lower()
            if "matr"  in cl: rename[c] = "matricula"
            elif "plan" in cl: rename[c] = "plano"
            elif "idade" in cl or cl == "idade": rename[c] = "idade"
        df.rename(columns=rename, inplace=True)

        # Encontra colunas de expostos/ocorridos por sexo
        rows = []
        for sexo in SEXOS:
            sx = sexo.lower()
            col_exp  = _find_col(df, [f"expostos_{sx}", f"exp_{sx}", f"expostos{sx}"])
            col_ocor = _find_col(df, [f"ocorridos_{sx}", f"ocor_{sx}", f"obitos_{sx}",
                                      f"ocorridos{sx}", f"obitos{sx}"])
            if col_exp is None or col_ocor is None:
                continue
            tmp = df[["matricula","plano","idade", col_exp, col_ocor]].copy()
            tmp.columns = ["matricula","plano","idade","expostos","ocorridos"]
            tmp["sexo"] = sexo
            rows.append(tmp)
        if not rows:
            st.error("Não foi possível identificar colunas de expostos/ocorridos por sexo.")
            return None
        df = pd.concat(rows, ignore_index=True)

    required = ["matricula","plano","sexo","idade","expostos","ocorridos"]
    missing = [r for r in required if r not in df.columns]
    if missing:
        st.error(f"Colunas não encontradas na planilha de população: {missing}")
        return None
    df["idade"]     = pd.to_numeric(df["idade"],     errors="coerce")
    df["expostos"]  = pd.to_numeric(df["expostos"],  errors="coerce").fillna(0)
    df["ocorridos"] = pd.to_numeric(df["ocorridos"], errors="coerce").fillna(0)
    return df[required]


@st.cache_data(show_spinner=False)
def carregar_tabuas(conteudo: bytes, nome: str) -> pd.DataFrame | None:
    """
    Formato wide:
      idade | AT2000_M | AT2000_F | BR_EMSsb_M | ...
    Retorna formato long: nome, sexo, idade, qx

    Convenção de nome de coluna: <NomeTábua>_<M|F>
    O separador pode ser _ ou espaço; o sufixo de sexo é a última parte.
    """
    buf = io.BytesIO(conteudo)
    df = pd.read_excel(buf) if nome.endswith(("xlsx", "xls")) else pd.read_csv(buf)
    df.columns = [str(c).strip() for c in df.columns]

    # Detecta se já está em formato long (tem coluna 'qx' ou 'nome')
    cols_low = [c.lower() for c in df.columns]
    if "qx" in cols_low or "nome" in cols_low:
        # Formato long legado
        rename = {}
        for c in df.columns:
            cl = c.lower()
            if "nom" in cl:   rename[c] = "nome"
            elif "sex" in cl: rename[c] = "sexo"
            elif "idade" in cl or cl == "id": rename[c] = "idade"
            elif "qx" in cl:  rename[c] = "qx"
        df.rename(columns=rename, inplace=True)
        df["sexo"] = df["sexo"].apply(_normalizar_sexo)
    else:
        # Formato wide novo
        col_idade = _find_col(df, ["idade","age","x"])
        if col_idade is None:
            st.error("Coluna 'idade' não encontrada na planilha de tábuas.")
            return None
        idades = pd.to_numeric(df[col_idade], errors="coerce")

        rows = []
        for col in df.columns:
            if col == col_idade:
                continue
            # Última parte após _ ou espaço é o sexo
            partes = col.replace(" ","_").split("_")
            sufixo = partes[-1].upper()
            sexo = _normalizar_sexo(sufixo)
            if sexo not in SEXOS:
                continue   # ignora colunas sem sufixo de sexo reconhecível
            nome_tabua = "_".join(partes[:-1])
            qx = pd.to_numeric(df[col], errors="coerce")
            tmp = pd.DataFrame({"nome": nome_tabua, "sexo": sexo,
                                 "idade": idades, "qx": qx})
            rows.append(tmp)

        if not rows:
            st.error("Nenhuma coluna de tábua reconhecida. Use o formato NomeTábua_M / NomeTábua_F.")
            return None
        df = pd.concat(rows, ignore_index=True)

    required = ["nome","sexo","idade","qx"]
    missing = [r for r in required if r not in df.columns]
    if missing:
        st.error(f"Colunas não encontradas na planilha de tábuas: {missing}")
        return None
    df["idade"] = pd.to_numeric(df["idade"], errors="coerce")
    df["qx"]    = pd.to_numeric(df["qx"],    errors="coerce")
    return df[required].dropna(subset=["idade","qx"])


def _normalizar_sexo(sexo_raw: object) -> str | None:
    if pd.isna(sexo_raw):
        return None
    sexo = str(sexo_raw).strip().upper()
    if sexo in ("M", "MASCULINO", "MASC", "MALE", "H", "HOMEM"):
        return "M"
    if sexo in ("F", "FEMININO", "FEM", "FEMALE", "MULHER"):
        return "F"
    return sexo

def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols_low = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_low:
            return cols_low[cand.lower()]
    return None


# ═══════════════════════════════════════════════════════════════════════════════
#  CÁLCULO BASE
# ═══════════════════════════════════════════════════════════════════════════════

def calcular_esperados(pop_df: pd.DataFrame, tabua_df: pd.DataFrame) -> pd.DataFrame:
    """Calcula os esperados de acordo com a tábua de mortalidade e a população."""
    merged = pop_df.merge(tabua_df[["idade","qx"]], on="idade", how="left")
    merged["esperados"] = merged["expostos"] * merged["qx"]
    return merged


# ═══════════════════════════════════════════════════════════════════════════════
#  AGRUPAMENTO DE IDADES (χ²)
# ═══════════════════════════════════════════════════════════════════════════════

def agrupar_idades(df: pd.DataFrame, min_esp: float) -> pd.DataFrame:
    """Agrupa as idades em grupos de acordo com o mínimo de esperados por grupo."""
    df = df.sort_values("idade").reset_index(drop=True)
    grupos, acc_o, acc_e, acc_n = [], 0.0, 0.0, 0.0
    idade_ini = float(df["idade"].iloc[0])

    for i, row in df.iterrows():
        acc_o += float(row["ocorridos"])
        acc_e += float(row["esperados"])
        acc_n += float(row["expostos"])
        if acc_e >= min_esp:
            grupos.append({"idade_ini": idade_ini, "idade_fim": float(row["idade"]),
                           "ocorridos": acc_o, "esperados": acc_e, "expostos": acc_n})
            acc_o = acc_e = acc_n = 0.0
            next_i = i + 1
            idade_ini = float(df["idade"].iloc[next_i]) if next_i < len(df) else None

    if acc_e > 0:
        if grupos:
            grupos[-1]["idade_fim"] = float(df["idade"].iloc[-1])
            grupos[-1]["ocorridos"] += acc_o
            grupos[-1]["esperados"] += acc_e
            grupos[-1]["expostos"]  += acc_n
        else:
            grupos.append({"idade_ini": float(df["idade"].iloc[0]),
                           "idade_fim": float(df["idade"].iloc[-1]),
                           "ocorridos": acc_o, "esperados": acc_e, "expostos": acc_n})
    if not grupos:
        return pd.DataFrame()
    result = pd.DataFrame(grupos)
    result.insert(0, "grupo", result.apply(
        lambda r: f"{int(r.idade_ini)}–{int(r.idade_fim)}", axis=1))
    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  TESTES ESTATÍSTICOS
# ═══════════════════════════════════════════════════════════════════════════════

def teste_qui_quadrado(df_merged, min_esp, alpha):
    """Teste de Qui-quadrado para verificar a adequação da tábua de mortalidade."""
    df_v = df_merged.dropna(subset=["qx"])
    df_v = df_v[df_v["esperados"] > 0]
    if df_v.empty:
        return np.nan, np.nan, np.nan, None, 0, pd.DataFrame()
    df_g = agrupar_idades(df_v, min_esp)
    if df_g.empty or len(df_g) < 2:
        return np.nan, np.nan, np.nan, None, len(df_g) if not df_g.empty else 0, df_g
    ocorridos, esperados = df_g["ocorridos"].values, df_g["esperados"].values
    gl   = len(df_g) - 1
    stat = float(np.sum((ocorridos - esperados)**2 / esperados))
    pval = float(1 - chi2.cdf(stat, df=gl))
    crit = float(chi2.ppf(1 - alpha, df=gl))
    return stat, pval, crit, bool(stat <= crit), len(df_g), df_g


def teste_ks(df_merged, alpha):
    """Teste de Kolmogorov-Smirnov para verificar a adequação da tábua de mortalidade."""
    df_v = df_merged.dropna(subset=["qx"]).sort_values("idade")
    ocorridos, esperados = df_v["ocorridos"].values.astype(float), df_v["esperados"].values.astype(float)
    tot_o, tot_e = ocorridos.sum(), esperados.sum()
    if tot_o == 0 or tot_e == 0:
        return np.nan, np.nan, np.nan, None
    cdf_o = np.cumsum(ocorridos) / tot_o
    cdf_e = np.cumsum(esperados) / tot_e
    stat  = float(np.max(np.abs(cdf_o - cdf_e)))
    n     = int(round(tot_o))
    crit  = float({0.10:1.22, 0.05:1.36, 0.01:1.63}.get(alpha, 1.36) / np.sqrt(n))
    lam   = stat * (np.sqrt(n) + 0.12 + 0.11/np.sqrt(n))
    pval  = float(max(0.0, min(1.0,
        2 * sum(((-1)**(k-1)) * np.exp(-2*k**2*lam**2) for k in range(1,101)))))
    return stat, pval, crit, bool(stat <= crit)


def teste_z_binomial(df_merged, alpha):
    """Bilateral: rejeita se |Z| > z_{α/2}."""
    df_v = df_merged.dropna(subset=["qx"])
    df_v = df_v[df_v["expostos"] > 0]
    if df_v.empty:
        return np.nan, np.nan, np.nan, None, None
    q    = df_v["qx"].clip(upper=0.9999)
    var  = (df_v["expostos"] * q * (1 - q)).sum()
    if var <= 0:
        return np.nan, np.nan, np.nan, None, None
    ocorridos_total, esperados_total = df_v["ocorridos"].sum(), df_v["esperados"].sum()
    z    = float((ocorridos_total - esperados_total) / np.sqrt(var))
    pval = float(2 * (1 - stats.norm.cdf(abs(z))))
    crit = float(stats.norm.ppf(1 - alpha/2))
    return z, pval, crit, bool(abs(z) <= crit), (1 if z >= 0 else -1)


# ═══════════════════════════════════════════════════════════════════════════════
#  ORQUESTRADOR
# ═══════════════════════════════════════════════════════════════════════════════

def _testar_merged(merged, min_esp, alpha_chi2, alpha_ks, alpha_z):
    """Recebe um DataFrame já com colunas 'ocorridos', 'esperados', 'qx', 'expostos', 'idade'
    e retorna (row_dict, df_grupos). Não faz nenhum join adicional."""
    merged = merged.copy()
    merged["esperados"] = merged["esperados"].fillna(0)
    # Garante que qx existe para os testes que precisam (Z usa expostos+qx)
    if "qx" not in merged.columns:
        merged["qx"] = np.nan

    q2, p2, c2, a2, n_g, df_g = teste_qui_quadrado(merged, min_esp, alpha_chi2)
    ks, pks, cks, aks          = teste_ks(merged, alpha_ks)
    z,  pz,  cz,  az, z_dir    = teste_z_binomial(merged, alpha_z)

    ocorridos_total = merged["ocorridos"].sum()
    esperados_total = merged["esperados"].sum()
    row = {
        "N Idades":    int(merged["idade"].nunique()),
        "N Grupos χ²": int(n_g),
        "Expostos":    float(merged["expostos"].sum()),
        "Ocorridos":   float(ocorridos_total),
        "Esperados":   float(esperados_total),
        "Razão O/E":   float(ocorridos_total/esperados_total) if esperados_total > 0 else np.nan,
        "χ² Stat": q2, "χ² GL":      int(n_g-1) if n_g >= 2 else np.nan,
        "χ² p-valor": p2, "χ² Crítico": c2, "χ² Aprovado": a2,
        "KS Stat": ks, "KS p-valor": pks, "KS Crítico": cks, "KS Aprovado": aks,
        "Z Stat":  z,  "Z p-valor":  pz,  "Z Crítico":  cz,  "Z Aprovado":  az,
        "Z Direção": ("O>E ↑" if z_dir==1 else "O<E ↓") if z_dir is not None else "—",
    }
    return row, df_g


def rodar_todos_testes(
    pop_df: pd.DataFrame,
    tabua_df: pd.DataFrame,
    alpha_chi2: float = 0.05,
    alpha_ks:   float = 0.05,
    alpha_z:    float = 0.05,
    min_esp_chi2: float = 5.0,
    fatores_agravo: dict | None = None,
):
    """
    Retorna (df_resultados, detalhes_grupos, merged_detalhes).
    'Sexo Pop.' pode ser 'M', 'F' ou 'Total'.
    """
    if fatores_agravo is None:
        fatores_agravo = {}

    resultados, detalhes_grupos, merged_detalhes = [], {}, {}

    for nome_tabua in tabua_df["nome"].unique():
        tb = tabua_df[tabua_df["nome"] == nome_tabua]
        for sexo_tabua in tb["sexo"].unique():
            fator = fatores_agravo.get((nome_tabua, sexo_tabua), 1.0)
            tb_sx = tb[tb["sexo"] == sexo_tabua].copy()
            tb_sx["qx"] = (tb_sx["qx"] * fator).clip(upper=1.0)

            merges_por_sexo = []   # acumula merged de cada sexo para o Total

            for sexo_pop in sorted(pop_df["sexo"].unique()):
                pop_sx = pop_df[pop_df["sexo"] == sexo_pop].copy()
                if pop_sx.empty:
                    continue

                # Calcula esperados e faz o join uma única vez
                mg = calcular_esperados(pop_sx, tb_sx)
                mg = mg.dropna(subset=["qx"])
                if mg.empty:
                    continue

                merges_por_sexo.append(mg)
                merged_detalhes[(nome_tabua, sexo_tabua, sexo_pop)] = mg

                row, df_g = _testar_merged(mg, min_esp_chi2, alpha_chi2, alpha_ks, alpha_z)
                detalhes_grupos[(nome_tabua, sexo_tabua, sexo_pop)] = df_g
                resultados.append({
                    "Tábua": nome_tabua, "Sexo Tábua": sexo_tabua,
                    "Sexo Pop.": sexo_pop, "Fator qx": fator, **row})

            # ── Agregado Total (M + F) ────────────────────────────────────────
            # Concatena os merged já calculados e soma por idade.
            # NÃO refaz join com a tábua — esperados já estão corretos.
            if len(merges_por_sexo) >= 1:
                mg_concat = pd.concat(merges_por_sexo, ignore_index=True)
                mg_total = (mg_concat
                            .groupby("idade", as_index=False)
                            .agg(
                                expostos =("expostos",  "sum"),
                                ocorridos=("ocorridos", "sum"),
                                esperados=("esperados", "sum"),
                                qx       =("qx",        "mean"),  # usado só no teste Z
                            ))
                merged_detalhes[(nome_tabua, sexo_tabua, "Total")] = mg_total
                row_t, df_g_t = _testar_merged(
                    mg_total, min_esp_chi2, alpha_chi2, alpha_ks, alpha_z)
                detalhes_grupos[(nome_tabua, sexo_tabua, "Total")] = df_g_t
                resultados.append({
                    "Tábua": nome_tabua, "Sexo Tábua": sexo_tabua,
                    "Sexo Pop.": "Total", "Fator qx": fator, **row_t})

    return pd.DataFrame(resultados), detalhes_grupos, merged_detalhes


# ═══════════════════════════════════════════════════════════════════════════════
#  GRÁFICOS
# ═══════════════════════════════════════════════════════════════════════════════

PAL = {"obs":"#2563eb","esp":"#f59e0b","oe":"#10b981",
       "red":"#ef4444","grid":"#e2e8f0","bg":"#f8fafc"}

def _style(ax, title="", xlabel="", ylabel=""):
    """Estilo dos gráficos."""
    ax.set_facecolor(PAL["bg"])
    ax.figure.patch.set_facecolor("white")
    ax.grid(axis="y", color=PAL["grid"], linewidth=0.8, zorder=0)
    ax.spines[["top","right"]].set_visible(False)
    ax.spines[["left","bottom"]].set_color("#cbd5e1")
    ax.tick_params(colors="#475569", labelsize=8)
    if title:
        ax.set_title(title,  fontsize=9.5, fontweight="bold", color="#0f172a", pad=7)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8, color="#64748b")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8, color="#64748b")


def grafico_oe_por_idade(merged, nome_tabua, sxt, sxp):
    """Gráfico de óbitos ocorridos e esperados por idade."""
    df = merged.dropna(subset=["qx"]).sort_values("idade")
    if df.empty: return None
    idades = df["idade"].astype(int).values
    ocorridos = df["ocorridos"].values
    esperados = df["esperados"].values
    oe = np.where(esperados > 0, ocorridos/esperados, np.nan)

    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(12,5.5),
        gridspec_kw={"height_ratios":[2.5,1],"hspace":0.35})
    ax1.bar(idades, esperados, width=0.65, color=PAL["esp"], alpha=0.80, label="Esperados", zorder=2)
    ax1.plot(idades, ocorridos, color=PAL["obs"], linewidth=2, marker="o",
             markersize=3, label="Ocorridos", zorder=3)
    _style(ax1, title=f"Ocorridos vs Esperados — {nome_tabua} ({sxt}) × Pop. ({sxp})",
           ylabel="Nº de óbitos")
    ax1.legend(fontsize=8, framealpha=0.6)

    ax2.axhline(1.0, color="#94a3b8", linewidth=1, linestyle="--")
    ax2.plot(idades, oe, color=PAL["oe"], linewidth=1.5, marker="o", markersize=3, zorder=3)
    ax2.fill_between(idades, 1, oe, where=(oe>=1), alpha=0.15, color=PAL["red"], interpolate=True)
    ax2.fill_between(idades, 1, oe, where=(oe<1),  alpha=0.15, color=PAL["obs"], interpolate=True)
    _style(ax2, ylabel="Razão O/E", xlabel="Idade")
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    plt.tight_layout()
    return fig


def grafico_cdf(merged, nome_tabua, sxt, sxp):
    """Gráfico de distribuição acumulada (CDF) dos óbitos e esperados."""
    df = merged.dropna(subset=["qx"]).sort_values("idade")
    if df.empty:
        return None
    ocorridos = df["ocorridos"].values.astype(float)
    esperados = df["esperados"].values.astype(float)
    total_ocorridos, total_esperados = ocorridos.sum(), esperados.sum()
    if total_ocorridos == 0 or total_esperados == 0:
        return None
    idades = df["idade"].astype(int).values
    cdf_o  = np.cumsum(ocorridos)/total_ocorridos
    cdf_e  = np.cumsum(esperados)/total_esperados
    diff   = np.abs(cdf_o-cdf_e)
    im     = np.argmax(diff)

    fig, ax = plt.subplots(figsize=(12,4))
    ax.step(idades, cdf_o, where="post", color=PAL["obs"], linewidth=2, label="CDF Ocorridos", zorder=3)
    ax.step(idades, cdf_e, where="post", color=PAL["esp"], linewidth=2, label="CDF Esperados", zorder=3)
    y_mid = (cdf_o[im]+cdf_e[im])/2
    ax.vlines(idades[im], min(cdf_o[im],cdf_e[im]), max(cdf_o[im],cdf_e[im]),
              color=PAL["red"], linewidth=1.5, linestyle="--", zorder=4)
    ax.annotate(f"D = {diff[im]:.4f}", xy=(idades[im], y_mid),
                xytext=(idades[im]+2, y_mid+0.06), fontsize=8, color=PAL["red"],
                arrowprops=dict(arrowstyle="->", color=PAL["red"]))
    ax.fill_between(idades, cdf_o, cdf_e, alpha=0.07, color=PAL["red"])
    _style(ax, title=f"CDF — {nome_tabua} ({sxt}) × Pop. ({sxp})",
           xlabel="Idade", ylabel="Prob. acumulada")
    ax.legend(fontsize=8, framealpha=0.6)
    plt.tight_layout()
    return fig


def grafico_chi2_grupos(df_g, stat, crit, nome_tabua, sxt, sxp):
    """Gráfico de contribuição ao χ² por grupo de idades."""
    if df_g is None or df_g.empty or len(df_g) < 2: return None
    ocorridos, esperados   = df_g["ocorridos"].values, df_g["esperados"].values
    contrib = np.where(esperados > 0, (ocorridos-esperados)**2/esperados, 0.0)
    limite  = crit/len(df_g)*2
    cores   = [PAL["red"] if c > limite else PAL["obs"] for c in contrib]
    fig, ax = plt.subplots(figsize=(12,4))
    ax.bar(df_g["grupo"].values, contrib, color=cores, alpha=0.85, zorder=3)
    ax.axhline(limite, color=PAL["red"], linewidth=1, linestyle="--",
               label=f"Limiar visual ≈ {limite:.2f}")
    _style(ax, title=f"Contribuição ao χ² — {nome_tabua} ({sxt}) × Pop. ({sxp})",
           xlabel="Grupo de idades", ylabel="(O−E)²/E")
    ax.set_xticklabels(df_g["grupo"].values, rotation=45, ha="right", fontsize=7.5)
    ax.legend(fontsize=8, framealpha=0.6)
    plt.tight_layout()
    return fig


def grafico_ranking_oe(df_res: pd.DataFrame):
    """Gráfico de ranking da razão O/E."""
    if df_res.empty: return None
    df     = df_res.copy().sort_values("Razão O/E")
    labels = df.apply(lambda r: f"{r['Tábua']} ({r['Sexo Tábua']}→{r['Sexo Pop.']})", axis=1)
    oe     = df["Razão O/E"].values
    cores  = [PAL["oe"] if 0.85<=v<=1.15 else PAL["red"] for v in oe]
    fig, ax = plt.subplots(figsize=(10, max(3, 0.46*len(df))))
    ax.scatter(oe, range(len(df)), color=cores, s=70, zorder=4)
    ax.axvline(1.0, color="#94a3b8", linewidth=1, linestyle="--")
    ax.axvspan(0.85, 1.15, alpha=0.07, color=PAL["oe"])
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(labels.values, fontsize=7.5)
    _style(ax, title="Ranking — Razão O/E", xlabel="Razão O/E")
    ax.legend(handles=[
        mpatches.Patch(color=PAL["oe"],  alpha=0.7, label="O/E ∈ [0,85;1,15]"),
        mpatches.Patch(color=PAL["red"], alpha=0.7, label="Fora da banda"),
    ], fontsize=8, framealpha=0.6)
    plt.tight_layout()
    return fig


def grafico_heatmap_aprovacao(df_res: pd.DataFrame):
    """Gráfico de mapa de aprovação dos testes."""
    if df_res.empty: return None
    testes = ["χ² Aprovado","KS Aprovado","Z Aprovado"]
    siglas = ["χ²","KS","Z"]
    df     = df_res.copy()
    df["label"] = df.apply(lambda r: f"{r['Tábua']} ({r['Sexo Tábua']}→{r['Sexo Pop.']})", axis=1)
    matrix = df[testes].applymap(lambda v: 1 if v is True else (0 if v is False else -1)).values
    cmap   = matplotlib.colors.ListedColormap(["#fee2e2","#d1fae5"])
    fig, ax = plt.subplots(figsize=(5, max(3, 0.46*len(df))))
    ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(3)); ax.set_xticklabels(siglas, fontsize=9, fontweight="bold")
    ax.set_yticks(range(len(df))); ax.set_yticklabels(df["label"].values, fontsize=7.5)
    ax.tick_params(left=False, bottom=False); ax.spines[:].set_visible(False)
    for i in range(len(df)):
        for j in range(3):
            v   = matrix[i,j]
            txt = "✓" if v==1 else ("✗" if v==0 else "—")
            cor = "#166534" if v==1 else ("#991b1b" if v==0 else "#64748b")
            ax.text(j, i, txt, ha="center", va="center", fontsize=11, color=cor, fontweight="bold")
    ax.set_title("Mapa de Aprovação", fontsize=9.5, fontweight="bold", color="#0f172a", pad=7)
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  EXPORTAÇÃO EXCEL — uma aba por tábua, 3 sub-seções (M / F / Total)
# ═══════════════════════════════════════════════════════════════════════════════

def _ap_str(v):
    return "Aprovado" if v is True else ("Reprovado" if v is False else "—")

def _fmt(v, decimais=4):
    if v is None or (isinstance(v,float) and np.isnan(v)): return "—"
    if isinstance(v, float): return round(v, decimais)
    return v


def gerar_excel(
    resultados: pd.DataFrame,
    detalhes_grupos: dict,
    merged_detalhes: dict,
) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:

        # ── Aba: Resumo geral ────────────────────────────────────────────────
        res = resultados.copy()
        for col in ("χ² Aprovado","KS Aprovado","Z Aprovado"):
            res[col] = res[col].apply(_ap_str)
        res.to_excel(writer, sheet_name="Resumo Geral", index=False)

        # ── Uma aba por tábua ────────────────────────────────────────────────
        tabuas = resultados["Tábua"].unique()
        for nome_tabua in tabuas:
            sheet_name = str(nome_tabua)[:31]
            rows_sheet  = []   # linhas que serão escritas

            for sexo_tabua in resultados.loc[
                    resultados["Tábua"]==nome_tabua, "Sexo Tábua"].unique():

                for sexo_pop in ("M","F","Total"):
                    key = (nome_tabua, sexo_tabua, sexo_pop)
                    mg  = merged_detalhes.get(key)
                    df_g = detalhes_grupos.get(key)
                    row_res = resultados[
                        (resultados["Tábua"]==nome_tabua) &
                        (resultados["Sexo Tábua"]==sexo_tabua) &
                        (resultados["Sexo Pop."]==sexo_pop)
                    ]

                    if mg is None or row_res.empty:
                        continue

                    r = row_res.iloc[0]

                    # Cabeçalho de seção
                    lbl = f"Tábua: {nome_tabua} ({sexo_tabua})  ×  População: {sexo_pop}"
                    rows_sheet.append({"A": lbl})
                    rows_sheet.append({})   # linha em branco

                    # Sumário estatístico
                    rows_sheet.append({"A":"Expostos","B": _fmt(r["Expostos"],1),
                                       "C":"Ocorridos","D":_fmt(r["Ocorridos"],1),
                                       "E":"Esperados","F":_fmt(r["Esperados"],4),
                                       "G":"Razão O/E","H":_fmt(r["Razão O/E"],4)})
                    rows_sheet.append({"A":"χ² Stat","B":_fmt(r["χ² Stat"]),
                                       "C":"χ² p-valor","D":_fmt(r["χ² p-valor"]),
                                       "E":"χ² Aprovado","F":_ap_str(r["χ² Aprovado"]),
                                       "G":"N Grupos","H":_fmt(r["N Grupos χ²"],0)})
                    rows_sheet.append({"A":"KS Stat","B":_fmt(r["KS Stat"]),
                                       "C":"KS p-valor","D":_fmt(r["KS p-valor"]),
                                       "E":"KS Aprovado","F":_ap_str(r["KS Aprovado"])})
                    rows_sheet.append({"A":"Z Stat","B":_fmt(r["Z Stat"]),
                                       "C":"Z p-valor","D":_fmt(r["Z p-valor"]),
                                       "E":"Z Aprovado","F":_ap_str(r["Z Aprovado"]),
                                       "G":"Z Direção","H":r.get("Z Direção","—")})
                    rows_sheet.append({})

                    # Tabela por idade
                    mg_out = mg[["idade","expostos","ocorridos","qx","esperados"]].copy()
                    mg_out.columns = ["Idade","Expostos","Ocorridos","qx","Esperados"]
                    mg_out["Razão O/E"] = np.where(
                        mg_out["Esperados"]>0,
                        mg_out["Ocorridos"]/mg_out["Esperados"], np.nan)
                    # header
                    rows_sheet.append({c:c for c in mg_out.columns})
                    for _, rw in mg_out.iterrows():
                        rows_sheet.append({c: _fmt(rw[c],4 if c in ("qx","Esperados","Razão O/E") else 1)
                                           for c in mg_out.columns})

                    # Tabela de grupos χ²
                    if df_g is not None and not df_g.empty:
                        rows_sheet.append({})
                        rows_sheet.append({"A":"Grupos χ²"})
                        df_g_out = df_g.copy()
                        df_g_out["(O-E)²/E"] = np.where(
                            df_g_out["esperados"]>0,
                            (df_g_out["ocorridos"]-df_g_out["esperados"])**2/df_g_out["esperados"], 0)
                        rows_sheet.append({c:c for c in df_g_out.columns})
                        for _, rw in df_g_out.iterrows():
                            rows_sheet.append(dict(rw))

                    rows_sheet.append({})
                    rows_sheet.append({"A": "─"*60})
                    rows_sheet.append({})

            if rows_sheet:
                # Converte para DataFrame com colunas A-H
                all_cols = set()
                for rw in rows_sheet:
                    all_cols.update(rw.keys())
                all_cols = sorted(all_cols, key=lambda c: (len(c), c))
                out_df = pd.DataFrame(rows_sheet, columns=all_cols)
                out_df.to_excel(writer, sheet_name=sheet_name, index=False)

    buf.seek(0)
    return buf.read()


# ═══════════════════════════════════════════════════════════════════════════════
#  FORMATAÇÃO DE TABELA NA INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════

def formatar_aprovado(val):
    """Formatação de aprovação dos testes."""
    return "✅ Aprovado" if val is True else ("❌ Reprovado" if val is False else "—")

def colorir_linha(row):
    """Colorir linha da tabela de resultados."""
    vals = [v for v in [row.get("χ² Aprovado"), row.get("KS Aprovado"), row.get("Z Aprovado")]
            if v is not None and not (isinstance(v,float) and np.isnan(v))]
    if not vals:      return [""]*len(row)
    if all(vals):     return ["background-color:#f0fdf4"]*len(row)
    if any(vals):     return ["background-color:#fffbeb"]*len(row)
    return ["background-color:#fef2f2"]*len(row)


# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown('<div class="main-title">⚖️ Aderência</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Tábuas de Mortalidade</div>', unsafe_allow_html=True)
    st.divider()

    st.markdown("#### 📂 Dados de entrada")
    arquivo_pop    = st.file_uploader("Planilha de Populações",  type=["xlsx","xls","csv"])
    arquivo_tabuas = st.file_uploader("Planilha de Tábuas",       type=["xlsx","xls","csv"])
    st.divider()

    st.markdown("#### ⚙️ Parâmetros dos testes")
    st.markdown('<span style="font-size:0.78rem;font-weight:600;color:#475569;">α por teste</span>',
                unsafe_allow_html=True)
    alpha_chi2 = st.select_slider("α — Qui-quadrado",
        options=[0.01,0.05,0.10], value=0.05,
        format_func=lambda x: f"{int(x*100)}%", key="a_chi2")
    alpha_ks   = st.select_slider("α — KS",
        options=[0.01,0.05,0.10], value=0.05,
        format_func=lambda x: f"{int(x*100)}%", key="a_ks")
    alpha_z    = st.select_slider("α — Teste Z  (bilateral)",
        options=[0.01,0.05,0.10], value=0.05,
        format_func=lambda x: f"{int(x*100)}%", key="a_z")
    min_esp_chi2 = st.number_input("Mín. esperados/grupo (χ²)",
        min_value=1.0, max_value=50.0, value=5.0, step=1.0)
    st.divider()
    st.markdown("""<div style="font-size:0.72rem;color:#94a3b8;line-height:1.75;">
    <b>Testes</b><br>
    • χ² — agrupamento dinâmico de idades<br>
    • KS — n = nº de óbitos observados<br>
    • Z — bilateral, α/2 em cada cauda
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="main-title">Aderência de Tábuas de Mortalidade</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Qui-quadrado com agrupamento dinâmico · Kolmogorov-Smirnov · Teste Z bilateral</div>',
            unsafe_allow_html=True)

# ── Tela de boas-vindas ──────────────────────────────────────────────────────
if arquivo_pop is None or arquivo_tabuas is None:
    st.markdown("""<div class="info-box">
    👈 <strong>Faça upload</strong> das planilhas no painel lateral para começar.
    </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-header">Formato — Populações</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({
            "matricula":["ENT001","ENT001"], "plano":["PLANO_A","PLANO_A"],
            "idade":[45,52],
            "expostos_M":[120.5,98.0], "ocorridos_M":[1,2],
            "expostos_F":[110.0,90.5], "ocorridos_F":[0,1],
        }), use_container_width=True, hide_index=True)
    with c2:
        st.markdown('<div class="section-header">Formato — Tábuas</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({
            "idade":[45,46,47],
            "AT2000_M":[0.003241,0.003512,0.003800],
            "AT2000_F":[0.002100,0.002280,0.002470],
            "BR_EMSsb_M":[0.004102,0.004430,0.004780],
        }), use_container_width=True, hide_index=True)
    st.stop()

# ── Carregamento ─────────────────────────────────────────────────────────────
# Lê conteúdo uma única vez e guarda — necessário para cache e para reset de agravo
_pop_bytes   = arquivo_pop.read();    arquivo_pop.seek(0)
_tabua_bytes = arquivo_tabuas.read(); arquivo_tabuas.seek(0)

pop_df   = carregar_populacao(_pop_bytes,   arquivo_pop.name)
tabua_df = carregar_tabuas(_tabua_bytes, arquivo_tabuas.name)

if pop_df is None or tabua_df is None:
    st.stop()

# ── Seleção de população ──────────────────────────────────────────────────────
st.markdown('<div class="section-header">Selecionar população para análise</div>',
            unsafe_allow_html=True)
col_ent, col_plan = st.columns(2)
with col_ent:
    entidades    = sorted(pop_df["matricula"].dropna().unique())
    entidade_sel = st.selectbox("Matrícula / Entidade", entidades)
with col_plan:
    planos       = sorted(pop_df[pop_df["matricula"]==entidade_sel]["plano"].dropna().unique())
    plano_sel    = st.selectbox("Plano de benefícios", planos)

pop_filtrada = pop_df[
    (pop_df["matricula"]==entidade_sel) & (pop_df["plano"]==plano_sel)].copy()

sexos_pop  = sorted(pop_filtrada["sexo"].dropna().unique())
total_exp  = pop_filtrada["expostos"].sum()
total_ocor = pop_filtrada["ocorridos"].sum()
n_idades   = pop_filtrada["idade"].nunique()
n_tabuas   = tabua_df["nome"].nunique()

c1,c2,c3,c4,c5 = st.columns(5)
for cw, lbl, val in zip([c1,c2,c3,c4,c5],
    ["Total Expostos","Total Ocorridos","Idades distintas","Sexos","Tábuas carregadas"],
    [f"{total_exp:,.1f}", f"{total_ocor:,.0f}", str(n_idades),
     " · ".join(sexos_pop), str(n_tabuas)]):
    cw.markdown(f'<div class="stat-card"><div class="label">{lbl}</div>'
                f'<div class="value">{val}</div></div>', unsafe_allow_html=True)

# ── Agravamento / Desagravamento por tábua ────────────────────────────────────
st.markdown('<div class="section-header">🔧 Agravamento / Desagravamento por tábua</div>',
            unsafe_allow_html=True)

combos_tabua = [
    (nome, sexo)
    for nome in sorted(tabua_df["nome"].unique())
    for sexo in sorted(tabua_df[tabua_df["nome"]==nome]["sexo"].unique())
]

# Chave de versão do arquivo de tábuas para invalidar session_state ao trocar arquivo
_tabua_hash = hash(_tabua_bytes)
if st.session_state.get("_tabua_hash") != _tabua_hash:
    # Novo arquivo carregado — limpa fatores anteriores
    for nome_t, sexo_t in combos_tabua:
        st.session_state.pop(f"fator_{nome_t}_{sexo_t}", None)
    st.session_state["_tabua_hash"] = _tabua_hash

col_exp, col_reset = st.columns([6,1])
with col_reset:
    if st.button("↺ Resetar", help="Volta todos os fatores para 1,00"):
        for nome_t, sexo_t in combos_tabua:
            st.session_state[f"fator_{nome_t}_{sexo_t}"] = 1.00
        st.rerun()

fatores_agravo: dict = {}
with col_exp:
    with st.expander("Ajustar fatores (1,00 = sem ajuste)", expanded=False):
        st.markdown(
            '<div style="font-size:0.78rem;color:#64748b;margin-bottom:0.6rem;">'
            "Multiplica o qx de cada tábua×sexo por um fator individual antes de calcular os esperados."
            "</div>", unsafe_allow_html=True)
        N_COLS = 4
        grid = st.columns(N_COLS)
        for idx, (nome_t, sexo_t) in enumerate(combos_tabua):
            CHAVE = f"fator_{nome_t}_{sexo_t}"
            with grid[idx % N_COLS]:
                fator = st.number_input(
                    f"{nome_t} ({sexo_t})",
                    min_value=0.10, max_value=5.00,
                    value=float(st.session_state.get(CHAVE, 1.00)),
                    step=0.01, format="%.2f", key=CHAVE)
            fatores_agravo[(nome_t, sexo_t)] = fator

        ativos = {k:v for k,v in fatores_agravo.items() if abs(v-1.0)>0.005}
        if ativos:
            partes = []
            for (n,s), v in sorted(ativos.items()):
                COR_HEX  = "#dc2626" if v>1 else "#2563eb"
                SINAL_SETA = "▲" if v>1 else "▼"
                partes.append(f'<span style="color:{COR_HEX};font-weight:600;">'
                              f'{SINAL_SETA} {nome_t} ({sexo_t}) × {v:.2f}</span>')
            st.markdown(
                '<div style="margin-top:0.4rem;font-size:0.8rem;line-height:2;">'
                "Ativos: " + " &nbsp;|&nbsp; ".join(partes) + "</div>", unsafe_allow_html=True)

# ── Executar testes ───────────────────────────────────────────────────────────
with st.spinner("Calculando testes de aderência..."):
    resultados, detalhes_grupos, merged_detalhes = rodar_todos_testes(
        pop_filtrada, tabua_df,
        alpha_chi2=alpha_chi2, alpha_ks=alpha_ks, alpha_z=alpha_z,
        min_esp_chi2=min_esp_chi2,
        fatores_agravo=fatores_agravo,
    )

if resultados.empty:
    st.warning("Nenhum resultado gerado. Verifique se as idades coincidem entre população e tábuas.")
    st.stop()

# ── Tabela de resultados ──────────────────────────────────────────────────────
st.markdown('<div class="section-header">Resultados dos testes</div>', unsafe_allow_html=True)

colf1, colf2, colf3 = st.columns(3)
with colf1:
    tab_filt = st.selectbox("Filtrar por tábua",
                            ["Todas"] + sorted(resultados["Tábua"].unique().tolist()))
with colf2:
    sxt_filt = st.selectbox("Sexo da tábua",
                            ["Todos"] + sorted(resultados["Sexo Tábua"].unique().tolist()))
with colf3:
    sxp_filt = st.selectbox("Sexo da população",
                            ["Todos"] + sorted(resultados["Sexo Pop."].unique().tolist()))

res_exibir = resultados.copy()
if tab_filt != "Todas":   res_exibir = res_exibir[res_exibir["Tábua"]      == tab_filt]
if sxt_filt != "Todos":   res_exibir = res_exibir[res_exibir["Sexo Tábua"] == sxt_filt]
if sxp_filt != "Todos":   res_exibir = res_exibir[res_exibir["Sexo Pop."]  == sxp_filt]

res_display = res_exibir.copy()
for col in ("χ² Aprovado","KS Aprovado","Z Aprovado"):
    res_display[col] = res_display[col].apply(formatar_aprovado)

float_fmt = {c:"{:.4f}" for c in [
    "Fator qx","Expostos","Ocorridos","Esperados","Razão O/E",
    "χ² Stat","χ² p-valor","χ² Crítico",
    "KS Stat","KS p-valor","KS Crítico",
    "Z Stat","Z p-valor","Z Crítico",
] if c in res_display.columns}

st.dataframe(
    res_display.style.apply(colorir_linha, axis=1).format(float_fmt, na_rep="—"),
    use_container_width=True, hide_index=True,
    height=min(40+38*len(res_display), 600),
)

# ── Resumo em cards ───────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Resumo por combinação tábua × sexo</div>',
            unsafe_allow_html=True)

card_cols = st.columns(4)
for i, (_, row) in enumerate(res_exibir.iterrows()):
    ap   = [v for v in [row["χ² Aprovado"],row["KS Aprovado"],row["Z Aprovado"]]
            if v is not None and not (isinstance(v,float) and np.isnan(v))]
    n_ap = sum(bool(a) for a in ap)
    COR  = "#16a34a" if n_ap==len(ap) else ("#d97706" if n_ap>0 else "#dc2626")
    EMOJI  = "✅" if n_ap==len(ap) else ("⚠️" if n_ap>0 else "❌")
    ng   = int(row["N Grupos χ²"]) if not (isinstance(row["N Grupos χ²"],float) and np.isnan(row["N Grupos χ²"])) else "—"
    def _b(v): return "✅" if v is True else ("❌" if v is False else "—")
    with card_cols[i%4]:
        st.markdown(f"""
        <div style="background:#f8fafc;border:1px solid #e2e8f0;border-top:3px solid {COR};
                    border-radius:8px;padding:0.8rem;margin-bottom:0.6rem;">
            <div style="font-weight:700;font-size:0.82rem;color:#0f172a;margin-bottom:0.2rem;">
                {EMOJI} {row['Tábua']}</div>
            <div style="font-size:0.7rem;color:#64748b;margin-bottom:0.4rem;">
                Tábua {row['Sexo Tábua']} × Pop. {row['Sexo Pop.']}
                &nbsp;|&nbsp; {ng} grupos χ²</div>
            <div style="font-size:0.76rem;color:#374151;">
                {_b(row['χ² Aprovado'])} χ² &nbsp;
                {_b(row['KS Aprovado'])} KS &nbsp;
                {_b(row['Z Aprovado'])} Z</div>
            <div style="font-size:0.7rem;color:#64748b;margin-top:0.3rem;">
                O/E: <b>{row['Razão O/E']:.4f}</b>
                &nbsp;·&nbsp; Fator: <b>{row['Fator qx']:.2f}</b></div>
        </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  GRÁFICOS
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-header">Análise gráfica</div>', unsafe_allow_html=True)

opcoes_combo = res_exibir.apply(
    lambda r: f"{r['Tábua']} ({r['Sexo Tábua']}) × Pop. ({r['Sexo Pop.']})", axis=1).tolist()

if opcoes_combo:
    SEL_COMBO = st.selectbox("Combinação (tábua × população)", opcoes_combo, key="combo_sel")
    ROW_COMBO = res_exibir.iloc[opcoes_combo.index(SEL_COMBO)]
else:
    SEL_COMBO = None
    ROW_COMBO = None

tab_vis, tab_chi, tab_cdf, tab_rank = st.tabs([
    "📊 Ocorridos vs Esperados", "🔢 Grupos χ²", "📈 Curvas KS", "🏆 Ranking O/E"])

with tab_vis:
    if ROW_COMBO is not None:
        mg = merged_detalhes.get((ROW_COMBO["Tábua"], ROW_COMBO["Sexo Tábua"], ROW_COMBO["Sexo Pop."]))
        if mg is not None:
            fig = grafico_oe_por_idade(mg, ROW_COMBO["Tábua"], ROW_COMBO["Sexo Tábua"], ROW_COMBO["Sexo Pop."])
            if fig: st.pyplot(fig, use_container_width=True); plt.close(fig)

with tab_chi:
    if ROW_COMBO is not None:
        df_g  = detalhes_grupos.get((ROW_COMBO["Tábua"], ROW_COMBO["Sexo Tábua"], ROW_COMBO["Sexo Pop."]), pd.DataFrame())
        ng    = len(df_g) if df_g is not None and not df_g.empty else 0
        st.markdown(f"""<div class="info-box">
        <b>{ng} grupos</b> · mínimo {min_esp_chi2:.0f} esperados/grupo.
        &nbsp; χ² = <b>{ROW_COMBO['χ² Stat']:.4f}</b> &nbsp;|&nbsp;
        Crítico = <b>{ROW_COMBO['χ² Crítico']:.4f}</b> &nbsp;|&nbsp;
        GL = <b>{int(ROW_COMBO['N Grupos χ²'])-1 if ROW_COMBO['N Grupos χ²'] and not np.isnan(ROW_COMBO['N Grupos χ²']) else '—'}</b>
        &nbsp;|&nbsp; p = <b>{ROW_COMBO['χ² p-valor']:.4f}</b>
        </div>""", unsafe_allow_html=True)
        if df_g is not None and not df_g.empty and len(df_g)>=2:
            col_t, col_g = st.columns([1,2])
            with col_t:
                df_show = df_g.copy()
                df_show["(O-E)²/E"] = np.where(df_show["esperados"]>0,
                    (df_show["ocorridos"]-df_show["esperados"])**2/df_show["esperados"],0)
                st.dataframe(df_show.style.format(
                    {"ocorridos":"{:.2f}","esperados":"{:.2f}",
                     "expostos":"{:.1f}","(O-E)²/E":"{:.4f}"}
                ), use_container_width=True, hide_index=True)
            with col_g:
                fig_c = grafico_chi2_grupos(df_g, ROW_COMBO["χ² Stat"], ROW_COMBO["χ² Crítico"],
                                            ROW_COMBO["Tábua"], ROW_COMBO["Sexo Tábua"], ROW_COMBO["Sexo Pop."])
                if fig_c: st.pyplot(fig_c, use_container_width=True); plt.close(fig_c)
        else:
            st.info("Grupos insuficientes para esta combinação.")

with tab_cdf:
    if ROW_COMBO is not None:
        mg_k  = merged_detalhes.get((ROW_COMBO["Tábua"], ROW_COMBO["Sexo Tábua"], ROW_COMBO["Sexo Pop."]))
        if mg_k is not None:
            fig_k = grafico_cdf(mg_k, ROW_COMBO["Tábua"], ROW_COMBO["Sexo Tábua"], ROW_COMBO["Sexo Pop."])
            if fig_k: st.pyplot(fig_k, use_container_width=True); plt.close(fig_k)

with tab_rank:
    col_r1, col_r2 = st.columns([1.6,1])
    with col_r1:
        fig_r = grafico_ranking_oe(res_exibir)
        if fig_r: st.pyplot(fig_r, use_container_width=True); plt.close(fig_r)
    with col_r2:
        fig_h = grafico_heatmap_aprovacao(res_exibir)
        if fig_h: st.pyplot(fig_h, use_container_width=True); plt.close(fig_h)

# ═══════════════════════════════════════════════════════════════════════════════
#  EXPORTAÇÃO
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-header">Exportar resultados</div>', unsafe_allow_html=True)

col_e1, col_e2 = st.columns(2)
with col_e1:
    excel_bytes = gerar_excel(resultados, detalhes_grupos, merged_detalhes)
    st.download_button(
        label="📥 Baixar planilha Excel",
        data=excel_bytes,
        file_name=f"aderencia_{entidade_sel}_{plano_sel}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
    st.caption("Abas: Resumo Geral + uma aba por tábua (M / F / Total por seção, com detalhe por idade e grupos χ²)")

with col_e2:
    if REPORTLAB_OK:
        # PDF simplificado — usa apenas df resumo
        try:
            buf_pdf = io.BytesIO()
            doc = SimpleDocTemplate(buf_pdf, pagesize=landscape(A4),
                                    rightMargin=1.5*cm, leftMargin=1.5*cm,
                                    topMargin=2*cm, bottomMargin=2*cm)
            sty = getSampleStyleSheet()
            ts  = ParagraphStyle("t", parent=sty["Title"], fontSize=13,
                                 textColor=rl_colors.HexColor("#0f172a"), spaceAfter=4)
            ss  = ParagraphStyle("s", parent=sty["Normal"], fontSize=9,
                                 textColor=rl_colors.HexColor("#64748b"), spaceAfter=3)
            hs  = ParagraphStyle("h", parent=sty["Heading2"], fontSize=10,
                                 textColor=rl_colors.HexColor("#1e40af"), spaceAfter=3)
            story = [
                Paragraph("Relatório de Aderência de Tábuas de Mortalidade", ts),
                Paragraph(f"Entidade: {entidade_sel}  |  Plano: {plano_sel}  |  "
                          f"α χ²={int(alpha_chi2*100)}%  KS={int(alpha_ks*100)}%  Z={int(alpha_z*100)}%", ss),
                HRFlowable(width="100%", thickness=1, color=rl_colors.HexColor("#e2e8f0")),
                Spacer(1, 0.3*cm), Paragraph("Resultados por Combinação", hs),
            ]
            cols_pdf = ["Tábua","Sexo Tábua","Sexo Pop.","Fator qx","Expostos",
                        "Ocorridos","Esperados","Razão O/E",
                        "χ² Stat","χ² p-valor","χ² Aprovado",
                        "KS Stat","KS p-valor","KS Aprovado",
                        "Z Stat","Z p-valor","Z Aprovado","Z Direção"]
            header = cols_pdf
            data_r = []
            for _, r in resultados.iterrows():
                row_p = []
                for c in cols_pdf:
                    v = r[c]
                    if c in ("χ² Aprovado","KS Aprovado","Z Aprovado"):
                        row_p.append("Aprov." if v is True else ("Reprov." if v is False else "—"))
                    elif isinstance(v,float) and np.isnan(v): row_p.append("—")
                    elif isinstance(v,float): row_p.append(f"{v:.4f}")
                    else: row_p.append(str(v))
                data_r.append(row_p)
            cw_pdf = [3.5*cm,1.3*cm,1.3*cm,1.5*cm,2*cm,
                      1.8*cm,1.8*cm,1.6*cm,
                      1.6*cm,1.8*cm,1.9*cm,
                      1.6*cm,1.8*cm,1.9*cm,
                      1.6*cm,1.8*cm,1.9*cm,1.5*cm]
            t = Table([header]+data_r, colWidths=cw_pdf, repeatRows=1)
            cmds = [
                ("BACKGROUND",(0,0),(-1,0),rl_colors.HexColor("#0f172a")),
                ("TEXTCOLOR", (0,0),(-1,0),rl_colors.white),
                ("FONTNAME",  (0,0),(-1,0),"Helvetica-Bold"),
                ("FONTSIZE",  (0,0),(-1,-1),6.5),
                ("ALIGN",     (0,0),(-1,-1),"CENTER"),
                ("VALIGN",    (0,0),(-1,-1),"MIDDLE"),
                ("BOX",       (0,0),(-1,-1),0.5,rl_colors.HexColor("#e2e8f0")),
                ("INNERGRID", (0,0),(-1,-1),0.25,rl_colors.HexColor("#e2e8f0")),
                ("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3),
            ]
            for i, row_p in enumerate(data_r, 1):
                n_ap = sum(row_p[cols_pdf.index(c)]=="Aprov."
                           for c in ("χ² Aprovado","KS Aprovado","Z Aprovado"))
                BACKGROUND_COLOR = ("#f0fdf4" if n_ap==3 else "#fef2f2" if n_ap==0 else "#fffbeb")
                cmds.append(("BACKGROUND",(0,i),(-1,i),rl_colors.HexColor(BACKGROUND_COLOR)))
            t.setStyle(TableStyle(cmds))
            story.append(t)
            doc.build(story)
            buf_pdf.seek(0)
            st.download_button(
                label="📄 Baixar relatório PDF",
                data=buf_pdf.read(),
                file_name=f"relatorio_aderencia_{entidade_sel}_{plano_sel}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception as exc:  # pylint: disable=broad-except
            st.warning(f"Erro ao gerar PDF: {exc}")
    else:
        st.info("Instale `reportlab` para habilitar o relatório PDF.")
