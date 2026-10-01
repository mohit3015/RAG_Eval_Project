"""
transcripts (.vtt)
      │
      ▼
  [ RETRIEVER ]  ──►  [ RERANKER ]  ──►  [ GENERATOR ]
   (embed+search)      (cross-encoder)     (LLM answer)
      │                                        │
      └──────────────► RAG PIPELINE ◄──────────┘
                     (retriever+reranker+generator combined)
                              │
                    ┌─────────┼─────────┐
                    ▼         ▼         ▼
               QUALITY    SAFETY    OPERATIONAL
              (correctness (scope,   (latency,
               style etc)  leakage,   cost,
                           toxicity) reliability)
                    │         │         │
                    └─────────┼─────────┘
                              ▼
                    evals/run_suite.py
                  (ek snapshot JSON banata hai)
                              │
                              ▼
                    evals/compare.py
              (baseline vs candidate compare karke
               PASS / REVIEW / FAIL verdict deta hai)

"""
              

"""
=======================================================================================
                                    1. Retriever
=======================================================================================

model:-
sentence-transformers/all-MiniLM-L6-v2 (HuggingFace, local)

kyun:-
Chhota, fast, free — koi API key nahi chahiye, chunks ko vectors mein badalta hai

Retriever ko akele test karta hai — kya wo relevant chunks nikal pa raha hai.

Metric:-	Kya check karta hai
Contextual Recall:-	Kya retriever ne saara zaroori context nikala (kuch chhoot to nahi gaya)
Contextual Precision:-	Kya relevant chunks top rank par aa rahe hain

"""


"""
======================================================================================
                                    2. Reranker
=======================================================================================

model:-
cross-encoder/ms-marco-MiniLM-L-6-v2 (HuggingFace, local)

kyun:-
Query+chunk ko saath padh kar accurate relevance score deta hai — embeddings se zyada precise

"""



"""
=======================================================================================
                    3. Generator LLM (final answer banane wala)
=======================================================================================

model:-
nvidia/nemotron-3-super-120b-a12b:free (via OpenRouter)

kyun:-
Free tier OpenRouter model — user ke actual question ka jawab deta hai

Generator ko isolation mein test karta hai — retriever ka output use nahi karte, 
seedha golden (known-good) context dete hain, taaki agar answer galat aaye to pata chale 
galti sirf generator ki hai, retriever ki nahi.

Metric:-	Kya check karta hai
Faithfulness:-	Kya generator ka answer sirf diye gaye context se bana, ya usne khud se facts invent kiye
Answer Relevancy:-	Kya answer actually question ka jawab deta hai

"""


# Judge LLM (evaluation ke liye)**************************

# model
# nvidia/nemotron-3-super-120b-a12b:free

# kyun>>>>>>>>>
# DeepEval ke metrics ko score/reason nikalne ke liye ek "judge" LLM chahiye — ye wahi kaam karta hai


"""
=======================================================================================
            4. Pipeline Level — Full Triad (eval_rag_pipeline.py)
=======================================================================================

Retriever + Reranker + Generator saath mein chained — real-world jaisa flow. 
Ab context bhi live retriever se aata hai, generator bhi live answer banata hai.

Metric:-	Kya check karta hai
Contextual Relevancy:-	Retrieved chunks kitne relevant hain (noise kitna hai)
Faithfulness:-	Generated answer context se grounded hai ya nahi
Answer Relevancy:-	Answer question ke liye relevant hai ya nahi

Ye teeno mila kar "RAG Triad" kehlata hai — sabse standard RAG evaluation pattern.
"""



"""
=======================================================================================
                5. Application Level (eval_application.py)
=======================================================================================

Poori application ko end-user experience ke hisaab se judge karta hai — GEval (custom rubric-based LLM judge) use karke.

Metric:-	Kya check karta hai
Correctness (GEval):-	Kya answer mein koi factual galti hai (completeness se alag)
Completeness (GEval):-	Kya answer ne expected answer ke saare key points cover kiye
Style (GEval):-	Kya answer CampusX jaisi teaching tone mein hai (conversational, intuitive) — ya dry/bullet-list

"""



"""
=======================================================================================
                                6. Safety Level
=======================================================================================

(eval_safety.py — merged from)
eval_scope.py
eval_leakage.py
eval_toxicity.py

Ye hard gates hain — koi bhi regression yahan release ko block kar deta hai.

Metric:-	Kya check karta hai
Scope Adherence (GEval):-	Bot apne teaching-assistant role mein rehta hai, ya jailbreak/roleplay se bhatak jaata hai
Leakage - Protected (GEval):-	System prompt / internal instructions / raw lecture transcripts leak to nahi ho rahe
Leakage - PII (PIILeakageMetric):-	Personal information (phone, email, ID) leak to nahi ho raha
Toxicity (ToxicityMetric, lower = safer)

"""



"""
=======================================================================================
                            7. Operational Level
=======================================================================================

(evals/eval_ops.py — merged from)
eval_latency.py
eval_cost.py
eval_reliability.py

Ye koi LLM judge use nahi karte — deterministic, code-based measurements hain.

Metric:-	Kya check karta hai
Latency - e2e p95:-	Poora answer milne mein kitna time lagta hai (95th percentile)
Latency - TTFT (info):-	Streaming mein pehla token kitni jaldi aata hai (perceived speed)
Cost - per query:-	Tokens × price se derive hota hai (free tier mein $0) 
Reliability - success rate:-	Kitne % requests successfully complete hue
Reliability - error rate:-	Kitne % requests fail hue (retries ke baad bhi)

"""



"""
=======================================================================================
        8. Regression Testing System (poore project ka "orchestration" layer)
=======================================================================================

harness.py:-	Shared helpers — load_goldens(), summarize_by_metric(), print_summary()

run_suite.py:-	Orchestrator — ek hi RagPipeline banata hai, 
                sabhi evals (component→pipeline→application→safety→ops) usi par chalata hai, 
                sab metrics ko ek dotted-ID snapshot mein flatten karta hai 
                (jaise retriever.contextual_recall.pass_rate), aur JSON file mein save karta hai
                
metric_registry.py:-	Har metric ID ke liye rule define karta hai — direction (higher/lower better), 
                        kind (gate/guardrail/info), tolerance (kitna change "noise" hai vs "real regression")
                        
compare.py:-	Do snapshots (baseline vs candidate) compare karta hai, verdict deta hai: PASS (safe),
                REVIEW (guardrail regress hua, human dekhe), FAIL (gate regress hua, block karo)

"""




"""
=======================================================================================
                                9. suite.py
=======================================================================================

run karne se pura pipeline test hota hai — component level se leke application level tak

    eval_retriever
    eval_generator
    eval_rag_pipeline
    eval_application
    eval_safety
    eval_ops

"""