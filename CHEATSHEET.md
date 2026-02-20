# 🚀 MLOps Quick Reference Cheatsheet

**Keep this handy during the training!**

---

## ⚡ Essential Commands

### Setup
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the App
```bash
# Terminal 1: Streamlit App
streamlit run streamlit_app.py

# Terminal 2: MLflow UI
mlflow ui
```

### Docker
```bash
# Build image
docker build -t mlops-iris:latest .

# Run container
docker run -p 8501:8501 mlops-iris:latest

# List containers
docker ps

# Stop container
docker stop <container_id>

# Remove image
docker rmi mlops-iris:latest
```

### Git & Deployment
```bash
# Initialize Git
git init
git add .
git commit -m "Initial commit"

# Push to GitHub
git remote add origin https://github.com/USERNAME/mlops-iris.git
git push -u origin main

# Make changes and update
git add .
git commit -m "Description of changes"
git push
```

---

## 🌐 URLs to Access

| Service | URL | Purpose |
|---------|-----|---------|
| Streamlit App | http://localhost:8501 | Main application |
| MLflow UI | http://localhost:5000 | Experiment tracking |
| Streamlit Cloud | https://share.streamlit.io | Deployment platform |
| API Docs | http://localhost:8501/docs | Swagger UI (if using FastAPI) |

---

## 📊 Hyperparameter Quick Guide

### n_estimators (Number of Trees)
- **Low (10-50):** Fast, may underfit
- **Medium (100-200):** Good balance
- **High (300-500):** Slow, may overfit

### max_depth (Tree Depth)
- **Shallow (1-3):** Simple model, may underfit
- **Medium (4-7):** Usually optimal
- **Deep (8+):** Complex, may overfit

### min_samples_split
- **Low (2-5):** More splits, complex model
- **High (10+):** Fewer splits, simpler model

### test_size
- **Small (10-20%):** More training data
- **Medium (20-30%):** Standard
- **Large (40-50%):** Better test reliability

---

## 🎯 Experiment Tracking Workflow

```
1. Adjust hyperparameters in Streamlit
   ↓
2. Click "Train Model"
   ↓
3. View results in Streamlit
   ↓
4. Check MLflow UI (refresh page)
   ↓
5. Compare experiments
   ↓
6. Select best model
```

---

## 🐛 Troubleshooting

### "Module not found"
```bash
pip install -r requirements.txt
```

### "Port already in use"
```bash
# Windows
netstat -ano | findstr :8501
taskkill /PID <PID> /F

# Mac/Linux
lsof -ti:8501 | xargs kill -9
```

### "Model not found" in app
1. Go to "Train Model" page
2. Train at least one model
3. Model auto-saves

### Docker build fails
```bash
# Check Docker is running
docker --version

# Clean rebuild
docker system prune -a
docker build --no-cache -t mlops-iris:latest .
```

### Git push rejected
```bash
git pull --rebase origin main
git push
```

---

## 📏 Example Measurements for Testing

| Species | Sepal Length | Sepal Width | Petal Length | Petal Width |
|---------|--------------|-------------|--------------|-------------|
| **Setosa** | 5.1 | 3.5 | 1.4 | 0.2 |
| **Versicolor** | 5.9 | 3.0 | 4.2 | 1.5 |
| **Virginica** | 6.5 | 3.0 | 5.5 | 2.0 |


## 💡 Pro Tips

✅ **Keep both terminals visible** - Streamlit + MLflow  
✅ **Refresh MLflow UI** after each training  
✅ **Document your experiments** - Add run names  
✅ **Start with defaults** - Then experiment  
✅ **Compare 3+ models** before choosing best  
✅ **Deploy early** - Get feedback fast  
✅ **Use version control** - Commit often  
✅ **Read error messages** - They're helpful!  

---

## 📚 Keyboard Shortcuts

### Streamlit
- `R` - Rerun app
- `Ctrl/Cmd + S` - Auto-rerun on file change
- `Ctrl/Cmd + C` - Stop server

### Terminal
- `Ctrl + C` - Stop running process
- `Ctrl + L` - Clear terminal
- `Tab` - Autocomplete

---

## 🔗 Important Links

- **MLflow Docs:** https://mlflow.org/docs/latest/
- **Streamlit Docs:** https://docs.streamlit.io
- **Docker Docs:** https://docs.docker.com
- **Scikit-learn:** https://scikit-learn.org
- **This Project:** [Your GitHub URL]

---

## 📝 Notes Space

Use this space for your own notes during training:

**Best Model Found:**
- n_estimators: ___
- max_depth: ___
- Accuracy: ___

**My Deployed URL:**
```
https://_____________________.streamlit.app
```

**Questions to Ask:**
1. ______________________________________
2. ______________________________________
3. ______________________________________

**Ideas for My Project:**
1. ______________________________________
2. ______________________________________
3. ______________________________________

---

## 🎯 Post-Training Action Items

- [ ] Save your deployed app URL
- [ ] Star the GitHub repo
- [ ] Try with your own dataset
- [ ] Share with colleagues
- [ ] Explore advanced MLflow features
- [ ] Join Streamlit community
- [ ] Build your own MLOps project!

---

**Keep learning! Keep deploying! 🚀**

*Need help? Ask the trainer or check the README.md for detailed instructions.*
