import json

# name, paradigm, mmlu(or None), comp, rev, compl, ord_, equiv, overall, ccs
rows = [
    ("Claude 4.5 Opus","Frontier dense",88.1, 0.8,5.8,3.2,0.0,0.0, 4.2, 0.2050),
    ("Claude 4.5 Sonnet","Frontier dense",85.4, 1.2,7.8,4.5,0.0,5.0, 5.9, 0.2110),
    ("o3-mini","Reasoning",88.2, 2.5,6.7,5.3,0.0,0.0, 5.4, 0.2178),
    ("GLM-5.2",None,None, 2.7,8.2,3.4,0.0,0.0, 5.7, 0.2230),
    ("Grok-3","Frontier dense",87.5, 1.6,10.5,6.2,1.5,15.0, 8.1, 0.2160),
    ("o4-mini","Reasoning",89.4, 3.1,10.3,6.2,0.0,12.5, 8.0, 0.2197),
    ("GPT-4.1","Frontier dense",90.1, 0.0,11.6,7.0,0.0,20.0, 8.8, 0.2156),
    ("GPT-4o","Frontier dense",88.7, 1.2,12.4,5.1,0.0,12.5, 8.9, 0.2173),
    ("DeepSeek-V3","Open MoE",84.0, 2.2,12.2,7.2,1.0,20.0, 9.1, 0.2165),
    ("Qwen3-235B","Open MoE",85.2, 2.1,12.0,7.1,0.0,40.0, 9.3, 0.2178),
    ("Gemini 2.0 Flash","Efficient",82.3, 2.0,13.5,7.5,2.0,12.5, 9.5, 0.2185),
    ("Mistral Large 2411","Open dense",82.6, 1.8,14.0,8.0,3.0,15.0, 10.2, 0.2155),
    ("Phi-4","Open small",78.9, 0.0,14.2,8.2,2.2,0.0, 10.5, 0.2179),
    ("DeepSeek R1","Open reasoning",86.1, 1.4,13.1,9.7,4.4,20.0, 10.9, 0.2171),
    ("Llama 3.3 70B","Open dense",83.7, 0.0,15.4,7.0,2.2,25.0, 11.4, 0.2131),
    ("Llama 4 Maverick","Open MoE",83.1, 2.4,14.2,12.6,8.9,7.1, 12.2, 0.2187),
    ("GPT-4o mini","Efficient",82.0, 0.0,17.9,10.1,13.3,12.5, 14.2, 0.2148),
]
rows.sort(key=lambda r: r[8])
with open("leaderboard.json","w") as f:
    json.dump(rows, f)
print(len(rows), "rows")
