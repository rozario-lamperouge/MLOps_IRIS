# 🌸 MLOps Iris Classifier - Complete Training Project

**A hands-on MLOps project for NIT scientists** covering experiment tracking, model deployment, containerization, and CI/CD.

---

## 📁 Project Structure (Just 4 Files!)

```
mlops-iris/
│
├── streamlit_app.py       # Main Streamlit application
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker containerization
├── .streamlit/
│   └── config.toml       # Streamlit configuration
└── README.md             # This file
```

**That's it!** Simple and powerful.

---

## 🚀 Quick Start (Local Development)

### Prerequisites

- Python 3.10+
- pip
- Docker Desktop 
- GitHub account

### Step 1: Clone & Setup

```bash
# Create project directory
mkdir mlops-iris
cd mlops-iris

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Run Streamlit App

```bash
streamlit run streamlit_app.py
```

🌐 Opens automatically at: **http://localhost:8501**

### Step 3: Start MLflow UI (In Separate Terminal)

```bash
# Keep first terminal running Streamlit
# Open NEW terminal and activate venv again:
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

# Start MLflow
mlflow ui
```

🌐 Opens at: **http://localhost:5000**

### 🎉 You're Ready!

Now you have:
- ✅ Streamlit app at **localhost:8501**
- ✅ MLflow dashboard at **localhost:5000**

**Keep both tabs open side by side!**

---

## 📚 How to Use the App

### 1️⃣ About Page
- Learn about the project
- Understand the dataset
- See the MLOps workflow

### 2️⃣ Train Model Page
- Adjust hyperparameters with sliders:
  - **n_estimators**: Number of trees (10-500)
  - **max_depth**: Tree depth (1-20)
  - **min_samples_split**: Minimum samples to split (2-20)
  - **test_size**: Test set percentage (10-50%)
- Click **"Train Model"**
- See results instantly
- Check confusion matrix

### 3️⃣ Make Predictions Page
- Enter flower measurements
- Get species prediction
- See confidence scores
- View probability distribution

### 4️⃣ MLflow Dashboard Page
- Instructions to access MLflow UI
- Tips for comparing experiments
- Understanding metrics

---

## 🧪 Training Exercises

### Exercise 1: Baseline Model
1. Go to "Train Model" page
2. Use default parameters
3. Click "Train Model"
4. Note the accuracy
5. Open MLflow UI (localhost:5000)
6. See your experiment logged

**Question:** What's the baseline accuracy?

### Exercise 2: Experiment with Hyperparameters

Train 5 different models:

| Experiment | n_estimators | max_depth | Expected Result |
|------------|--------------|-----------|-----------------|
| 1 (baseline) | 100 | 5 | Good balance |
| 2 | 10 | 3 | Underfitting? |
| 3 | 500 | 10 | Overfitting? |
| 4 | 200 | 7 | Better accuracy? |
| 5 | 150 | 4 | Your choice! |

**Questions:**
- Which model has the best accuracy?
- Which is fastest to train?
- What's the trade-off?

### Exercise 3: MLflow Analysis

In MLflow UI:
1. Click "Experiments" → "iris-classification"
2. Select multiple runs (checkbox)
3. Click "Compare"
4. Sort by accuracy (click column header)
5. Export best model

**Questions:**
- Can you find the best model quickly?
- What patterns do you see?
- How does max_depth affect accuracy?

### Exercise 4: Make Predictions

Test these measurements:

**Setosa (should predict Setosa):**
- Sepal Length: 5.1, Width: 3.5
- Petal Length: 1.4, Width: 0.2

**Versicolor (should predict Versicolor):**
- Sepal Length: 5.9, Width: 3.0
- Petal Length: 4.2, Width: 1.5

**Edge case (uncertain prediction):**
- Sepal Length: 6.0, Width: 3.0
- Petal Length: 4.0, Width: 1.3

**Questions:**
- How confident is the model?
- Which predictions are uncertain?

---

## 🐳 Docker Deployment

### Build Docker Image

```bash
docker build -t mlops-iris:latest .
```

### Run Container

```bash
docker run -p 8501:8501 mlops-iris:latest
```

🌐 Access at: **http://localhost:8501**

### Run with Volume (Persist MLflow logs)

```bash
docker run -p 8501:8501 -v $(pwd)/mlruns:/app/mlruns mlops-iris:latest
```

**On Windows PowerShell:**
```bash
docker run -p 8501:8501 -v ${PWD}/mlruns:/app/mlruns mlops-iris:latest
```

### Stop Container

```bash
# Find container ID
docker ps

# Stop container
docker stop <container_id>
```

---

## 🌐 Deploy to Streamlit Cloud

### Step 1: Push to GitHub

```bash
# Initialize git
git init

# Add files
git add .

# Commit
git commit -m "Initial commit: MLOps Iris Classifier"

# Create GitHub repo (on github.com)
# Then link and push:
git remote add origin https://github.com/YOUR_USERNAME/mlops-iris.git
git branch -M main
git push -u origin main
```

### Step 2: Deploy on Streamlit Cloud

1. Go to **https://share.streamlit.io**
2. Click **"New app"**
3. Connect your GitHub account
4. Select:
   - Repository: `YOUR_USERNAME/mlops-iris`
   - Branch: `main`
   - Main file: `streamlit_app.py`
5. Click **"Deploy"**

⏱️ **Deployment takes 2-3 minutes**

### Step 3: Share Your App

Once deployed, you get a URL like:
```
https://YOUR_USERNAME-mlops-iris-streamlit-app-RANDOM.streamlit.app
```

### 🔄 Auto-Deployment (CI/CD Magic!)

**Every time you push to GitHub, Streamlit Cloud automatically:**
1. Detects the change
2. Rebuilds the app
3. Redeploys automatically
4. Updates the live URL

**Example workflow:**
```bash

# Make changes to streamlit_app.py
vim streamlit_app.py

# Commit and push
git add streamlit_app.py
git commit -m "Improved UI"
git push

# Streamlit Cloud automatically redeploys!
# No manual steps needed!
```

**This is CI/CD in action!** 🚀

---

## 🎓 Key MLOps Concepts Explained

### 1. Experiment Tracking (MLflow)

**Problem:** How do you remember which hyperparameters gave the best results?

**Solution:** MLflow automatically logs:
- Parameters (what you changed)
- Metrics (accuracy, F1 score)
- Models (the trained model itself)
- Artifacts (confusion matrix, etc.)

**Benefits:**
- ✅ Reproducibility - Can recreate any experiment
- ✅ Comparison - See all runs side by side
- ✅ Collaboration - Share results with team
- ✅ Governance - Track model lineage

### 2. Model Serving (Streamlit)

**Problem:** How do you make your model accessible to non-technical users?

**Solution:** Streamlit creates a web interface where anyone can:
- Input data
- Get predictions
- No coding required!

**Benefits:**
- ✅ User-friendly interface
- ✅ Real-time predictions
- ✅ Easy to share (just a URL)
- ✅ Interactive exploration

### 3. Containerization (Docker)

**Problem:** "It works on my machine!" - Different environments cause issues.

**Solution:** Docker packages:
- Your code
- Dependencies
- Python version
- Everything needed to run

**Benefits:**
- ✅ Reproducibility - Same environment everywhere
- ✅ Portability - Runs on any system with Docker
- ✅ Isolation - Doesn't interfere with other projects
- ✅ Scalability - Easy to deploy multiple instances

### 4. CI/CD (Streamlit Cloud Auto-Deploy)

**Problem:** Manual deployment is slow and error-prone.

**Solution:** Automatic deployment pipeline:
```
Code Change → Git Push → Auto Build → Auto Deploy → Live Update
```

**Benefits:**
- ✅ Speed - Changes go live in minutes
- ✅ Reliability - Consistent deployment process
- ✅ Efficiency - No manual steps
- ✅ Rollback - Easy to revert if needed

---

## 📊 Understanding the Metrics

### Accuracy
```
Accuracy = (Correct Predictions) / (Total Predictions)
```
- **Good for:** Balanced datasets
- **Range:** 0.0 to 1.0 (higher is better)
- **Target:** > 0.95 for this dataset

### F1 Score
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```
- **Good for:** Imbalanced datasets
- **Range:** 0.0 to 1.0 (higher is better)
- **What it means:** Balance between precision and recall

### Confusion Matrix

```
                Predicted
               S    Ve   Vi
Actual    S  [10    0    0]
          Ve [ 0    9    1]
          Vi [ 0    1    9]
```

- **Diagonal:** Correct predictions
- **Off-diagonal:** Errors
- **Look for:** High numbers on diagonal, low elsewhere

---

## 🔧 Troubleshooting

### Issue: "Module not found"

**Solution:**
```bash
# Make sure venv is activated
pip install -r requirements.txt
```

### Issue: "Port 8501 already in use"

**Solution:**
```bash
# Windows
netstat -ano | findstr :8501
taskkill /PID <PID> /F

# Mac/Linux
lsof -ti:8501 | xargs kill -9

# Or use different port
streamlit run streamlit_app.py --server.port=8502
```

### Issue: "No model found" in Predictions page

**Solution:**
1. Go to "Train Model" page
2. Train at least one model
3. Model will be saved automatically

### Issue: MLflow UI shows no experiments

**Solution:**
```bash
# Make sure you're in the right directory
ls mlruns  # Should exist

# If empty, train a model first
```

### Issue: Docker build fails

**Solution:**
```bash
# Check Docker is running
docker --version

# Clear Docker cache
docker system prune -a

# Rebuild
docker build -t mlops-iris:latest .
```

### Issue: Streamlit Cloud deployment fails

**Solution:**
1. Check `requirements.txt` is in root directory
2. Make sure `streamlit_app.py` is in root (not in subfolder)
3. Verify all imports are in `requirements.txt`
4. Check deployment logs on Streamlit Cloud

---

## 🎨 Customization Ideas

### For Your Own Projects:

1. **Change the Dataset:**
   ```python
   # Replace load_iris() with your data
   df = pd.read_csv("your_data.csv")
   X = df.drop('target', axis=1)
   y = df['target']
   ```

2. **Try Different Models:**
   ```python
   from sklearn.svm import SVC
   from sklearn.neural_network import MLPClassifier
   from sklearn.ensemble import GradientBoostingClassifier
   ```

3. **Add More Features:**
   - Feature importance plots
   - ROC curves
   - Learning curves
   - Cross-validation

4. **Enhance UI:**
   - Add file upload for batch predictions
   - Include data visualization
   - Add model comparison page
   - Export predictions to CSV

---

## 📚 Resources & Next Steps

### Documentation:
- **Streamlit:** https://docs.streamlit.io
- **MLflow:** https://mlflow.org/docs/latest/index.html
- **Docker:** https://docs.docker.com
- **Scikit-learn:** https://scikit-learn.org

### Advanced Topics to Explore:

1. **Data Versioning:** DVC (Data Version Control)
2. **Model Monitoring:** Evidently AI, WhyLabs
3. **Feature Stores:** Feast, Tecton
4. **Advanced MLflow:** Model Registry, Model Serving
5. **Production Deployment:** AWS SageMaker, Google Vertex AI, Azure ML
6. **Kubernetes:** For scalable deployment
7. **A/B Testing:** Test multiple models in production
8. **AutoML:** Automated hyperparameter tuning


## 🤝 Contributing & Feedback

This is a training project, but you can:
- ⭐ Star the repo if you found it helpful
- 🐛 Report issues or bugs
- 💡 Suggest improvements
- 📝 Share your own extensions

---

### Before Implimenting:

- [ ] Python 3.10+ installed
- [ ] pip working
- [ ] Docker Desktop installed
- [ ] GitHub account created
- [ ] Internet connection verified
- [ ] Code editor ready (VS Code recommended)


### Post-Implimentation:

- [ ] Customize with own dataset
- [ ] Share deployed app with colleagues
- [ ] Explore advanced features
- [ ] Plan next MLOps project

---

## 🎯 Success Criteria

By the end of this implimentation, you should be able to:

✅ **Train** machine learning models with different hyperparameters  
✅ **Track** experiments systematically using MLflow  
✅ **Compare** model performance across multiple runs  
✅ **Deploy** models as web applications  
✅ **Containerize** applications with Docker  
✅ **Share** your work via public URLs  
✅ **Understand** the complete MLOps workflow  
✅ **Apply** these concepts to your own research  

---

## 💡 Final Tips

1. **Start Simple:** Get the basic workflow working first
2. **Experiment Freely:** MLflow tracks everything, so try things!
3. **Document:** Add notes in MLflow for each run
4. **Share Early:** Deploy and get feedback
5. **Iterate:** Improve based on what you learn
6. **Have Fun:** MLOps doesn't have to be intimidating!

---

## 🌟 What Makes This Project Special

✅ **Production-Ready:** Not a toy example, real patterns  
✅ **Educational:** Clear explanations at every step  
✅ **Minimal:** Just 4 files, no overwhelming structure  
✅ **Complete:** Covers entire MLOps workflow  
✅ **Practical:** Immediately applicable to your work  
✅ **Shareable:** Deploy and show your colleagues  
✅ **Extensible:** Easy to adapt for your use cases  

---

**Ready to become an MLOps practitioner? Let's go! 🚀**