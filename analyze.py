from main import analyze, analyze_batch

files = ["Main.java", "app.py", "server.go"]

print(analyze_batch(files))