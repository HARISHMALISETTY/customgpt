# filters.py

def is_software_question(text: str) -> bool:
    keywords = [
        "python", "java", "javascript", "react", "node", "html", "css",
        "sql", "mysql", "mongodb", "database", "api", "rest", "json",
        "algorithm", "data structure", "dsa", "oops", "class", "object",
        "function", "variable", "loop", "if", "else", "for", "while",
        "devops", "docker", "kubernetes", "cloud", "aws", "azure", "gcp",
        "machine learning", "ml", "ai", "neural", "model", "framework",
        "django", "flask", "express", "spring", "angular", "reactjs",
        "git", "github", "version control", "server", "hosting", "deployment",
        "bug", "error", "exception", "stack trace"
    ]

    text = text.lower()
    return any(word in text for word in keywords)
