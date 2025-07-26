import os
import json
import streamlit as st
from datetime import datetime
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from pypdf import PdfReader
from crewai import Crew, Agent, Task, Process

def display_salary_report(result):
    try:
        if isinstance(result, str):
            clean_result = result.replace('"raw";\n', '').replace('"pydantic": NULL\n', '').replace('"json_dict": NULL\n', '').strip()
            try:
                data = json.loads(clean_result)
            except json.JSONDecodeError:
                salary_part = clean_result.split('"salary_range":')[1].split('}')[0] + '}' if '"salary_range":' in clean_result else '{}'
                data = {"salary_range": json.loads(salary_part)}
        else:
            data = result

        with st.container(border=True):
            st.subheader("💰 Salary Estimation Report")
            st.write(f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}*")

            st.markdown("### 📊 Salary Range (Annual)")
            min_sal = data.get('salary_range', {}).get('min', 0)
            max_sal = data.get('salary_range', {}).get('max', 0)
            med_sal = data.get('salary_range', {}).get('median', 0)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(label="Minimum", value=f"₹{min_sal:,}", delta="Base")
            with col2:
                st.metric(label="Median", value=f"₹{med_sal:,}", delta="Typical")
            with col3:
                st.metric(label="Maximum", value=f"₹{max_sal:,}", delta="Top")

            if max_sal > min_sal:
                progress_value = (med_sal - min_sal) / (max_sal - min_sal)
                st.progress(progress_value)
            st.caption(f"Salary range from ₹{min_sal:,} to ₹{max_sal:,}")

            st.markdown("### 🌍 Market Comparison")
            st.info(data.get('market_comparison', "Comparison data not available"))

            confidence = data.get('confidence', 'Medium').lower()
            confidence_map = {
                'high': ('✅ High Confidence', 'green'),
                'medium': ('⚠️ Medium Confidence', 'orange'),
                'low': ('❌ Low Confidence', 'red')
            }
            conf_text, conf_color = confidence_map.get(confidence, ('⚠️ Medium Confidence', 'orange'))
            st.markdown(f"### 🔍 Confidence Level")
            st.markdown(f'<span style="color:{conf_color}">{conf_text}</span>', unsafe_allow_html=True)

            st.markdown("### 📈 Influencing Factors")
            factors = data.get('factors', [])
            if factors:
                for factor in factors:
                    st.markdown(f"- {factor}")
            else:
                st.warning("No specific factors identified")

    except Exception as e:
        st.code(result)

st.set_page_config(page_title="AI Salary Benchmarking", layout="centered")
st.title("📄 AI Resume Salary Benchmarking")
st.markdown("Upload your resume and get an **estimated salary** based on your experience and skills.")

os.environ["GROQ_API_KEY"] = "your api key"
os.environ["GROK_API_KEY"]=st.secrets["GROQ_API_KEY"]

llm = ChatGroq(
    model_name="groq/llama3-70b-8192",
    temperature=0.7
)

uploaded_file = st.file_uploader("Upload your resume (PDF)", type="pdf")

if uploaded_file:
    try:
        pdf_reader = PdfReader(uploaded_file)
        resume_text = ""
        for page in pdf_reader.pages:
            resume_text += page.extract_text() or ""

        if not resume_text:
            st.error("❌ Could not extract text. Try a different PDF.")
            st.stop()

        with st.expander("📝 Extracted Resume Text"):
            st.text(resume_text[:5000] + ("..." if len(resume_text) > 5000 else ""))

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(resume_text)

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_texts(chunks, embeddings)

        resume_analyst = Agent(
            role="Senior Resume Analyst",
            goal="Extract accurate job titles, skills, experience duration, location, and education from resumes",
            backstory="""You are a top-tier HR professional with 15+ years of experience in technical recruiting.
            You specialize in parsing complex resumes and extracting precise information about candidates' qualifications.""",
            llm=llm,
            verbose=True,
            allow_delegation=False
        )

        salary_expert = Agent(
            role="Senior Compensation Analyst",
            goal="Provide market-accurate salary estimates based on experience, skills, and location",
            backstory="""You are a certified compensation specialist with access to the latest salary surveys from
            Payscale, Glassdoor, and Indeed. You specialize in Indian tech market salaries.""",
            llm=llm,
            verbose=True,
            allow_delegation=False
        )

        extract_info_task = Task(
            description="""Analyze the resume thoroughly and extract:
            - All job titles held
            - Total years of professional experience
            - Top 10 technical skills
            - Current location (city/country)
            - Educational qualifications""",
            expected_output="""{
                "job_titles": ["position1", "position2"],
                "total_experience": "X years",
                "skills": ["skill1", "skill2"],
                "location": "City, Country",
                "education": ["Degree1", "Degree2"]
            }""",
            agent=resume_analyst,
            output_file="resume_data.json"
        )

        estimate_salary_task = Task(
            description="""Using the extracted resume data:
            1. Calculate appropriate salary range for Indian market
            2. Consider experience, skills, and location
            3. Compare with market benchmarks
            4. Explain key influencing factors""",
            expected_output="""{
                "salary_range": {"min": X, "max": Y, "median": Z},
                "currency": "INR",
                "market_comparison": "Compared to similar roles in [location]",
                "factors": ["experience", "skills", "location"],
                "confidence": "High/Medium/Low"
            }""",
            agent=salary_expert,
            context=[extract_info_task],
            output_file="salary_report.json"
        )

        crew = Crew(
            agents=[resume_analyst, salary_expert],
            tasks=[extract_info_task, estimate_salary_task],
            verbose=True,
            process=Process.sequential
        )

        if st.button("🚀 Estimate Salary"):
            with st.spinner("🔍 Analyzing resume and calculating salary..."):
                try:
                    result = crew.kickoff()
                    st.success("✅ Salary Estimated Successfully!")
                    display_salary_report(result)

                except Exception as e:
                    st.error(f"❌ Error during analysis: {str(e)}")

    except Exception as e:
        st.error(f"❌ Failed to process PDF: {str(e)}")
