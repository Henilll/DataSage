from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats as scipy_stats
import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional
import warnings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
warnings.filterwarnings('ignore')

# LangChain imports - supporting multiple providers
try:
    from langchain_groq import ChatGroq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

try:
    from langchain_community.llms import HuggingFaceHub
    from langchain.chains import LLMChain
    from langchain.prompts import PromptTemplate
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

try:
    from langchain_community.llms import Ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

app = FastAPI(
    title="DataSage API",
    description="AI-powered data analytics platform with FREE AI models",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "../static")
PLOT_DIR = os.path.join(STATIC_DIR, "plots")

os.makedirs(PLOT_DIR, exist_ok=True)

# In-memory storage for datasets
datasets_store = {}


class ChatRequest(BaseModel):
    session_id: str
    question: str


def calculate_quality_score(df: pd.DataFrame) -> Dict:
    """Calculate comprehensive data quality metrics"""
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isnull().sum().sum()
    
    completeness = ((total_cells - missing_cells) / total_cells * 100)
    
    numeric_df = df.select_dtypes(include=np.number)
    if not numeric_df.empty:
        valid_cells = total_cells - missing_cells - np.isinf(numeric_df).sum().sum()
    else:
        valid_cells = total_cells - missing_cells
    validity = (valid_cells / total_cells * 100)
    
    unique_rows = df.drop_duplicates().shape[0]
    consistency = (unique_rows / df.shape[0] * 100)
    
    quality_score = (completeness * 0.5 + validity * 0.3 + consistency * 0.2)
    
    return {
        "quality_score": round(quality_score, 1),
        "completeness": round(completeness, 1),
        "validity": round(validity, 1),
        "consistency": round(consistency, 1)
    }


def detect_data_types(df: pd.DataFrame) -> Dict:
    """Detect and categorize column data types"""
    types = {
        "numeric": 0,
        "text": 0,
        "datetime": 0,
        "boolean": 0
    }
    
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            types["numeric"] += 1
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            types["datetime"] += 1
        elif pd.api.types.is_bool_dtype(df[col]):
            types["boolean"] += 1
        else:
            types["text"] += 1
    
    return types


def calculate_statistics(df: pd.DataFrame) -> Dict:
    """Calculate detailed statistics for numeric columns"""
    stats_dict = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols[:5]:
        col_data = df[col].dropna()
        
        if len(col_data) == 0:
            continue
        
        Q1 = col_data.quantile(0.25)
        Q3 = col_data.quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((col_data < (Q1 - 1.5 * IQR)) | (col_data > (Q3 + 1.5 * IQR))).sum()
        
        stats_dict[col] = {
            "mean": float(col_data.mean()),
            "median": float(col_data.median()),
            "std": float(col_data.std()),
            "min": float(col_data.min()),
            "max": float(col_data.max()),
            "outliers": int(outliers)
        }
    
    return stats_dict


def find_correlations(df: pd.DataFrame, threshold: float = 0.7) -> List[Dict]:
    """Find strong correlations between numeric columns"""
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.shape[1] < 2:
        return []
    
    corr_matrix = numeric_df.corr()
    correlations = []
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr_value = corr_matrix.iloc[i, j]
            if abs(corr_value) >= threshold:
                correlations.append({
                    "var1": corr_matrix.columns[i],
                    "var2": corr_matrix.columns[j],
                    "score": round(corr_value, 2)
                })
    
    return sorted(correlations, key=lambda x: abs(x["score"]), reverse=True)[:6]


def generate_insights(df: pd.DataFrame, statistics: Dict, correlations: List) -> List[str]:
    """Generate AI-powered insights from the data"""
    insights = []
    
    missing = df.isnull().sum()
    if missing.sum() > 0:
        worst_col = missing.idxmax()
        worst_pct = (missing[worst_col] / len(df)) * 100
        if worst_pct > 20:
            insights.append(
                f"⚠️ Critical: '{worst_col}' has {worst_pct:.1f}% missing values. "
                f"Consider imputation or removal strategies."
            )
        elif worst_pct > 5:
            insights.append(
                f"📊 '{worst_col}' has {worst_pct:.1f}% missing values. "
                f"This may impact analysis quality."
            )
    
    for col, stat in statistics.items():
        if stat['outliers'] > 0:
            outlier_pct = (stat['outliers'] / len(df)) * 100
            insights.append(
                f"🔍 '{col}' contains {stat['outliers']} outliers ({outlier_pct:.1f}%). "
                f"Range: {stat['min']:.2f} to {stat['max']:.2f}, Mean: {stat['mean']:.2f}"
            )
        
        col_data = df[col].dropna()
        if len(col_data) > 3:
            skewness = scipy_stats.skew(col_data)
            if abs(skewness) > 1:
                direction = "right" if skewness > 0 else "left"
                insights.append(
                    f"📈 '{col}' shows significant {direction}-skew (skewness: {skewness:.2f}). "
                    f"Consider transformation for modeling."
                )
    
    for corr in correlations[:3]:
        strength = "very strong" if abs(corr['score']) > 0.9 else "strong"
        direction = "positive" if corr['score'] > 0 else "negative"
        insights.append(
            f"🔗 {strength.capitalize()} {direction} correlation ({corr['score']}) detected "
            f"between '{corr['var1']}' and '{corr['var2']}'."
        )
    
    if len(df) < 100:
        insights.append(
            "⚡ Small dataset detected. Results may have limited statistical power. "
            "Consider collecting more data for robust analysis."
        )
    
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        dup_pct = (duplicates / len(df)) * 100
        insights.append(
            f"⚠️ Found {duplicates} duplicate rows ({dup_pct:.1f}% of data). "
            f"Consider deduplication to improve data quality."
        )
    
    return insights[:8] if insights else ["✨ Data looks clean! No major issues detected."]


def create_visualizations(df: pd.DataFrame) -> List[str]:
    """Generate comprehensive visualizations"""
    charts = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        return charts
    
    colors = ['#667eea', '#764ba2', '#f093fb', '#4facfe']
    
    for i, col in enumerate(numeric_cols[:4]):
        try:
            fig = px.histogram(
                df,
                x=col,
                nbins=30,
                title=f"Distribution: {col.replace('_', ' ').title()}",
                template="plotly_white",
                color_discrete_sequence=[colors[i % len(colors)]]
            )
            
            fig.update_layout(
                font=dict(family="Inter, sans-serif", size=12),
                title_font=dict(size=16, family="Inter, sans-serif"),
                showlegend=False,
                height=350
            )
            
            filename = f"{col}_{uuid.uuid4().hex[:8]}.png"
            filepath = os.path.join(PLOT_DIR, filename)
            fig.write_image(filepath, width=600, height=350)
            charts.append(f"plots/{filename}")
        except Exception as e:
            print(f"Error creating histogram for {col}: {e}")
            continue
    
    if len(numeric_cols) >= 2:
        try:
            corr_matrix = df[numeric_cols].corr()
            
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=np.round(corr_matrix.values, 2),
                texttemplate='%{text}',
                textfont={"size": 10},
                colorbar=dict(title="Correlation")
            ))
            
            fig.update_layout(
                title="Correlation Heatmap",
                template="plotly_white",
                font=dict(family="Inter, sans-serif", size=11),
                title_font=dict(size=16, family="Inter, sans-serif"),
                height=400,
                width=500
            )
            
            filename = f"correlation_{uuid.uuid4().hex[:8]}.png"
            filepath = os.path.join(PLOT_DIR, filename)
            fig.write_image(filepath, width=500, height=400)
            charts.append(f"plots/{filename}")
        except Exception as e:
            print(f"Error creating correlation heatmap: {e}")
    
    if len(numeric_cols) >= 1:
        try:
            sample_col = numeric_cols[0]
            fig = px.box(
                df,
                y=sample_col,
                title=f"Outlier Detection: {sample_col.replace('_', ' ').title()}",
                template="plotly_white",
                color_discrete_sequence=['#667eea']
            )
            
            fig.update_layout(
                font=dict(family="Inter, sans-serif", size=12),
                title_font=dict(size=16, family="Inter, sans-serif"),
                showlegend=False,
                height=350
            )
            
            filename = f"boxplot_{uuid.uuid4().hex[:8]}.png"
            filepath = os.path.join(PLOT_DIR, filename)
            fig.write_image(filepath, width=600, height=350)
            charts.append(f"plots/{filename}")
        except Exception as e:
            print(f"Error creating box plot: {e}")
    
    return charts


def prepare_minimal_context(df: pd.DataFrame, question: str) -> str:
    context = []
    context.append(f"Rows: {len(df)}, Columns: {len(df.columns)}")
    context.append(f"Columns: {', '.join(df.columns)}")

    q = question.lower()

    if "missing" in q:
        missing = df.isnull().sum()
        context.append("Missing Values:")
        for col in missing[missing > 0].index:
            context.append(f"{col}: {missing[col]}")

    if "correlation" in q:
        numeric_cols = df.select_dtypes(include=np.number).columns
        context.append(f"Numeric Columns: {', '.join(numeric_cols)}")

    if "average" in q or "mean" in q:
        for col in df.select_dtypes(include=np.number).columns[:5]:
            context.append(f"{col} mean: {df[col].mean():.2f}")

    return "\n".join(context)


def answer_with_groq(question: str, df: pd.DataFrame) -> str:
    """Use Groq for fast, free inference"""
    try:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return None
        
        llm = ChatGroq(
            # model="llama-3.1-70b-versatile",  # Fast and free!
            model="llama-3.1-8b-instant",

            groq_api_key=api_key,
            temperature=0,
            max_tokens=1000
        )
        
        data_context = prepare_minimal_context(df, question)

        
        prompt = f"""You are DataSage AI, an expert data analyst. Analyze this dataset and answer the user's question.

Dataset Information:
{data_context}

User Question: {question}

Provide a clear, concise answer with relevant statistics. Use emojis (📊 📈 🔍 💡) to make it engaging."""

        response = llm.invoke(prompt)
        return response.content
    
    except Exception as e:
        print(f"Groq error: {e}")
        return None


def answer_with_huggingface(question: str, df: pd.DataFrame) -> str:
    """Use Hugging Face for free inference"""
    try:
        api_key = os.getenv("HUGGINGFACE_API_KEY")
        if not api_key:
            return None
        
        llm = HuggingFaceHub(
            repo_id="microsoft/Phi-3-mini-4k-instruct",  # Fast, free model
            huggingfacehub_api_token=api_key,
            model_kwargs={"temperature": 0.1, "max_length": 500}
        )
        
        data_context = prepare_data_context(df)
        
        prompt = f"""You are a data analyst. Answer this question about the dataset:

Dataset: {df.shape[0]} rows, {df.shape[1]} columns
Columns: {', '.join(df.columns.tolist())}

Question: {question}

Answer briefly and clearly:"""

        response = llm.invoke(prompt)
        return response
    
    except Exception as e:
        print(f"Hugging Face error: {e}")
        return None


def answer_with_ollama(question: str, df: pd.DataFrame) -> str:
    """Use local Ollama for completely free, private inference"""
    try:
        llm = Ollama(
            model="llama3.2",  # or "mistral", "phi3", etc.
            temperature=0
        )
        
        data_context = prepare_data_context(df)
        
        prompt = f"""You are DataSage AI, a data analyst assistant.

Dataset Info:
{data_context}

Question: {question}

Provide a clear answer with statistics. Use emojis for emphasis."""

        response = llm.invoke(prompt)
        return response
    
    except Exception as e:
        print(f"Ollama error: {e}")
        return None


def answer_question_with_ai(question: str, df: pd.DataFrame) -> str:
    # 1️⃣ Rule-based first for simple analytics
    simple_answer = answer_question_fallback(question, df)
    if "🤖 I can help you analyze" not in simple_answer:
        return simple_answer

    # 2️⃣ Then AI (complex questions only)
    if GROQ_AVAILABLE:
        response = answer_with_groq(question, df)
        if response and len(response.strip()) > 30:
            return response

    if OLLAMA_AVAILABLE:
        response = answer_with_ollama(question, df)
        if response:
            return response

    if HF_AVAILABLE:
        response = answer_with_huggingface(question, df)
        if response:
            return response

    return simple_answer



def answer_question_fallback(question: str, df: pd.DataFrame) -> str:
    """Fallback rule-based question answering"""
    
    question_lower = question.lower()
    
    if "how many" in question_lower and ("row" in question_lower or "record" in question_lower):
        return f"📊 The dataset contains **{len(df):,} rows** (records)."
    
    if "how many" in question_lower and "column" in question_lower:
        return f"📋 The dataset has **{len(df.columns)} columns**: {', '.join(df.columns.tolist())}"
    
    if "missing" in question_lower or "null" in question_lower:
        missing = df.isnull().sum()
        if missing.sum() == 0:
            return "✨ Great news! There are **no missing values** in this dataset."
        else:
            result = "🔍 Missing values found:\n\n"
            for col in missing[missing > 0].index:
                pct = (missing[col] / len(df)) * 100
                result += f"- **{col}**: {missing[col]:,} missing ({pct:.1f}%)\n"
            return result
    
    if "average" in question_lower or "mean" in question_lower:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            result = "📊 **Average (Mean) values**:\n\n"
            for col in numeric_cols:
                result += f"- **{col}**: {df[col].mean():.2f}\n"
            return result
        else:
            return "⚠️ No numeric columns found to calculate averages."
    
    if "max" in question_lower or "maximum" in question_lower:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            result = "📈 **Maximum values**:\n\n"
            for col in numeric_cols:
                result += f"- **{col}**: {df[col].max():.2f}\n"
            return result
    
    if "min" in question_lower or "minimum" in question_lower:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            result = "📉 **Minimum values**:\n\n"
            for col in numeric_cols:
                result += f"- **{col}**: {df[col].min():.2f}\n"
            return result
    
    if "correlation" in question_lower:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 2:
            corr_matrix = df[numeric_cols].corr()
            result = "🔗 **Strong correlations** (|r| > 0.7):\n\n"
            found = False
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.7:
                        found = True
                        result += f"- **{corr_matrix.columns[i]}** ↔ **{corr_matrix.columns[j]}**: {corr_val:.2f}\n"
            return result if found else "No strong correlations (|r| > 0.7) found."
        else:
            return "Need at least 2 numeric columns to calculate correlations."
    
    if "columns" in question_lower or "fields" in question_lower:
        result = f"📋 **Dataset Columns** ({len(df.columns)} total):\n\n"
        for col in df.columns:
            result += f"- **{col}** ({df[col].dtype})\n"
        return result
    
    if "data type" in question_lower or "dtype" in question_lower:
        result = "🏷️ **Data Types**:\n\n"
        for col in df.columns:
            result += f"- **{col}**: {df[col].dtype}\n"
        return result
    
    if "summary" in question_lower or "overview" in question_lower:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        result = f"📊 **Dataset Summary**:\n\n"
        result += f"- **Rows**: {len(df):,}\n"
        result += f"- **Columns**: {len(df.columns)}\n"
        result += f"- **Numeric Columns**: {len(numeric_cols)}\n"
        result += f"- **Missing Values**: {df.isnull().sum().sum():,}\n"
        if len(numeric_cols) > 0:
            result += f"\n**Numeric Statistics**:\n"
            for col in numeric_cols[:3]:
                result += f"- **{col}**: Mean={df[col].mean():.2f}, Std={df[col].std():.2f}\n"
        return result
    
    # Default response with AI provider status
    providers = []
    if GROQ_AVAILABLE and os.getenv("GROQ_API_KEY"):
        providers.append("✅ Groq (Fast & Free)")
    elif GROQ_AVAILABLE:
        providers.append("⚠️ Groq (Add API key)")
    
    if OLLAMA_AVAILABLE:
        providers.append("✅ Ollama (Local)")
    
    if HF_AVAILABLE and os.getenv("HUGGINGFACE_API_KEY"):
        providers.append("✅ Hugging Face")
    elif HF_AVAILABLE:
        providers.append("⚠️ Hugging Face (Add API key)")
    
    status = "\n".join(providers) if providers else "⚠️ No AI providers configured"
    
    return (
        "🤖 I can help you analyze your data! Try asking:\n\n"
        "- How many rows are in the dataset?\n"
        "- What are the column names?\n"
        "- Are there any missing values?\n"
        "- What's the average of [column]?\n"
        "- Show me correlations\n"
        "- Give me a summary\n\n"
        # f"**AI Status:**\n{status}\n\n"
        # "💡 **Tip**: Add free API keys in .env for AI-powered responses!"
    )


@app.get("/")
async def root():
    """API health check"""
    ai_status = {
        "groq": GROQ_AVAILABLE and bool(os.getenv("GROQ_API_KEY")),
        "huggingface": HF_AVAILABLE and bool(os.getenv("HUGGINGFACE_API_KEY")),
        "ollama": OLLAMA_AVAILABLE
    }
    
    return {
        "status": "operational",
        "service": "DataSage API",
        "version": "2.0.0",
        "features": ["analytics", "visualizations", "ai_chatbot"],
        "ai_providers": ai_status
    }


@app.post("/analyze")
async def analyze_csv(file: UploadFile = File(...)):
    """Comprehensive CSV analysis endpoint"""
    
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    
    try:
        df = pd.read_csv(file.file)
        
        if df.empty:
            raise HTTPException(status_code=400, detail="CSV file is empty")
        
        session_id = str(uuid.uuid4())
        
        datasets_store[session_id] = {
            "dataframe": df,
            "timestamp": datetime.now()
        }
        
        quality_metrics = calculate_quality_score(df)
        data_types = detect_data_types(df)
        statistics = calculate_statistics(df)
        correlations = find_correlations(df)
        charts = create_visualizations(df)
        insights = generate_insights(df, statistics, correlations)
        
        memory_usage = df.memory_usage(deep=True).sum()
        memory_mb = memory_usage / (1024 ** 2)
        memory_str = f"{memory_mb:.2f} MB" if memory_mb >= 1 else f"{memory_usage / 1024:.2f} KB"
        
        missing_dict = df.isnull().sum().to_dict()
        missing_dict = {k: int(v) for k, v in missing_dict.items() if v > 0}
        total_missing = sum(missing_dict.values())
        missing_percentage = (total_missing / (df.shape[0] * df.shape[1])) * 100 if df.shape[0] > 0 else 0
        
        response = {
            **quality_metrics,
            "summary": {
                "rows": int(df.shape[0]),
                "columns": int(df.shape[1]),
                "missing": missing_dict
            },
            "types": data_types,
            "statistics": statistics,
            "correlations": correlations,
            "charts": charts,
            "insights": insights,
            "total_missing": total_missing,
            "missing_percentage": round(missing_percentage, 2),
            "memory_usage": memory_str,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "session_id": session_id
        }
        
        return response
    
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="CSV file is empty or corrupted")
    except pd.errors.ParserError as e:
        raise HTTPException(status_code=400, detail=f"CSV parsing error: {str(e)}")
    except Exception as e:
        print(f"Analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")


@app.post("/chat")
async def chat_with_data(chat_request: ChatRequest):
    """AI Chatbot endpoint with FREE models"""
    
    if chat_request.session_id not in datasets_store:
        raise HTTPException(
            status_code=404,
            detail="Session not found. Please upload a file first."
        )
    
    try:
        df = datasets_store[chat_request.session_id]["dataframe"]
        
        # Try AI providers, fallback to rule-based
        answer = answer_question_with_ai(chat_request.question, df)
        
        return {
            "question": chat_request.question,
            "answer": answer,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    except Exception as e:
        print(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")


@app.delete("/cleanup")
async def cleanup_plots():
    """Clean up old plot files"""
    try:
        count = 0
        for filename in os.listdir(PLOT_DIR):
            if filename.endswith('.png'):
                os.remove(os.path.join(PLOT_DIR, filename))
                count += 1
        
        current_time = datetime.now()
        sessions_to_remove = []
        for session_id, data in datasets_store.items():
            if (current_time - data["timestamp"]).seconds > 3600:
                sessions_to_remove.append(session_id)
        
        for session_id in sessions_to_remove:
            del datasets_store[session_id]
        
        return {
            "status": "success",
            "files_deleted": count,
            "sessions_cleaned": len(sessions_to_remove)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cleanup error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    print("=" * 70)
    print("🚀 DataSage API Server")
    print("=" * 70)
    print("📍 API: http://127.0.0.1:8000")
    print("📚 Docs: http://127.0.0.1:8000/docs")
    print("🤖 AI Chatbot: FREE Models Enabled")
    print("-" * 70)
    print("AI Providers Available:")
    if GROQ_AVAILABLE:
        status = "✅ Configured" if os.getenv("GROQ_API_KEY") else "⚠️  Add API Key"
        print(f"  • Groq (Fast & Free): {status}")
    if OLLAMA_AVAILABLE:
        print(f"  • Ollama (Local): ✅ Ready")
    if HF_AVAILABLE:
        status = "✅ Configured" if os.getenv("HUGGINGFACE_API_KEY") else "⚠️  Add API Key"
        print(f"  • Hugging Face: {status}")
    print("=" * 70)
    uvicorn.run(app, host="0.0.0.0", port=8000)