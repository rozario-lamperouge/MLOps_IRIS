# 🌸 MLOps Iris Classifier - Complete Package


---

## 📦 What's Included

This package contains a complete, working MLOps project with:
- ✅ Streamlit app for interactive ML training
- ✅ MLflow integration for experiment tracking  
- ✅ Docker containerization
- ✅ Auto-deployment to Streamlit Cloud (CI/CD)
- ✅ Complete documentation and guides

---

## 📁 Files in This Package

### 🎯 Essential Files (Required)

| File | Purpose | Who Needs It |
|------|---------|--------------|
| **streamlit_app.py** | Main application | Everyone |
| **requirements.txt** | Python dependencies | Everyone |
| **Dockerfile** | Container definition | Everyone |
| **.gitignore** | Git ignore rules | Everyone |
| **.streamlit/config.toml** | Streamlit config | Everyone |

### 📖 Documentation Files

| File | Purpose | Who Needs It |
|------|---------|--------------|
| **README.md** | Participant guide & quick start | Participants |
| **TRAINER_GUIDE.md** | Session plan & talking points | Trainer only |
| **CHEATSHEET.md** | Quick reference | Participants |

---

## 🚀 Quick Start for Trainers

### Before the Session:

1. **Test the setup:**
```bash
# Create project directory
mkdir mlops-iris
cd mlops-iris

# Copy all files to this directory
# Install and test
pip install -r requirements.txt
streamlit run streamlit_app.py
mlflow ui  # in separate terminal
```

2. **Read TRAINER_GUIDE.md** - Your detailed session plan

3. **Prepare:**
   - Test Docker build
   - Create demo GitHub repo
   - Have Streamlit Cloud account ready

### During the Session:

- Follow **TRAINER_GUIDE.md** for minute-by-minute plan
- Give participants **CHEATSHEET.md** for quick reference
- Point them to **README.md** for detailed instructions

---

## 📚 Quick Start for Participants

### What You'll Build:

By the end of 4 hours, you'll have:
1. ✅ A trained ML model with tracked experiments
2. ✅ Interactive web app for predictions
3. ✅ Dockerized application
4. ✅ Live deployed app URL you can share!

### Steps:

1. **Setup**
   - Create virtual environment
   - Install dependencies from `requirements.txt`
   - Run `streamlit_app.py`

2. **Train Models*
   - Use Streamlit UI to train models
   - Track experiments with MLflow
   - Compare results

3. **Containerize**
   - Build Docker image from `Dockerfile`
   - Run containerized app

4. **Deploy**
   - Push to GitHub
   - Deploy to Streamlit Cloud
   - Get shareable URL!

**See README.md for detailed instructions!**

---

## 🎯 Key Features

### 1. Interactive Training Interface
- Sliders for hyperparameters (n_estimators, max_depth, etc.)
- Real-time training with visual feedback
- Confusion matrix visualization
- Multiple pages: About, Train, Predict, MLflow

### 2. Automatic Experiment Tracking
- Every model training logged to MLflow
- Parameters, metrics, and models saved
- Easy comparison in MLflow UI
- No manual logging needed!

### 3. User-Friendly Predictions
- Simple input form for flower measurements
- Instant predictions with confidence scores
- Probability visualization
- Example measurements provided

### 4. Production-Ready Deployment
- Docker containerization for reproducibility
- One-click deployment to Streamlit Cloud
- Automatic CI/CD (push to GitHub → auto-deploy)
- Shareable public URL

---

## 🎓 Learning Outcomes

After this training, participants will be able to:

✅ Train ML models with experiment tracking  
✅ Use MLflow to compare model performance  
✅ Build interactive ML applications  
✅ Containerize applications with Docker  
✅ Deploy ML apps to the cloud  
✅ Implement basic CI/CD pipelines  
✅ Apply these skills to their own research  

---

## 💡 Why This Approach?

### Compared to API-based approach:
✅ **More engaging** - Visual, interactive training  
✅ **Easier to understand** - Everything in one app  
✅ **Better for demos** - Non-technical users can use it  
✅ **Accessible** - Everyone gets a link  

### Compared to complex MLOps platforms:
✅ **Simple** - Only 5 core files  
✅ **Understandable** - Clear, readable code  
✅ **Practical** - Realistic production patterns  
✅ **Scalable** - Easy to extend  

---

## 🔄 The Complete Workflow

```
┌─────────────────────────────────────────────────┐
│  1. Train Model (Streamlit UI)                  │
│     - Adjust hyperparameters                    │
│     - Click "Train"                             │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│  2. Automatic MLflow Logging                    │
│     - Parameters saved                          │
│     - Metrics recorded                          │
│     - Model artifacts stored                    │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│  3. Compare in MLflow UI                        │
│     - View all experiments                      │
│     - Sort by accuracy                          │
│     - Select best model                         │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│  4. Make Predictions (Streamlit UI)             │
│     - Input measurements                        │
│     - Get instant predictions                   │
│     - See confidence scores                     │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│  5. Containerize (Docker)                       │
│     - Build image                               │
│     - Ensure reproducibility                    │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│  6. Deploy (Streamlit Cloud)                    │
│     - Push to GitHub                            │
│     - Auto-deploy (CI/CD)                       │
│     - Share URL                                 │
└─────────────────────────────────────────────────┘
```


## 🛠️ Technical Stack

| Component | Technology | Why? |
|-----------|-----------|------|
| **UI Framework** | Streamlit | Simple, interactive, perfect for ML |
| **ML Library** | Scikit-learn | Industry standard, beginner-friendly |
| **Experiment Tracking** | MLflow | Open-source, feature-rich |
| **Visualization** | Plotly | Interactive charts |
| **Containerization** | Docker | Industry standard |
| **Deployment** | Streamlit Cloud | Free, easy, auto-CI/CD |
| **Version Control** | Git/GitHub | Standard practice |

---

## 📊 Dataset Information

**Iris Flower Dataset:**
- **Samples:** 150 (50 per species)
- **Features:** 4 (sepal/petal length/width)
- **Classes:** 3 (Setosa, Versicolor, Virginica)
- **Type:** Multiclass classification
- **Difficulty:** Beginner-friendly
- **Why this dataset?** Classic ML benchmark, easy to understand

---

## 🔧 Customization Guide

### For Trainers:

**Easy customizations:**
- Change color scheme in `.streamlit/config.toml`
- Add your institution logo
- Modify page titles in `streamlit_app.py`

**Advanced customizations:**
- Add more ML models (SVM, Neural Networks)
- Include cross-validation
- Add feature importance plots
- Implement A/B testing

### For Participants (After Training):

**Replace the dataset:**
```python
# Instead of:
from sklearn.datasets import load_iris
X, y = load_iris(return_X_y=True)

# Use your data:
import pandas as pd
df = pd.read_csv("your_data.csv")
X = df.drop('target', axis=1)
y = df['target']
```

**Try different models:**
```python
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
# Replace RandomForestClassifier with these
```

---

## 📚 Additional Resources

### Documentation:
- **Streamlit:** https://docs.streamlit.io
- **MLflow:** https://mlflow.org/docs/latest/
- **Docker:** https://docs.docker.com
- **Scikit-learn:** https://scikit-learn.org

### Tutorials:
- MLflow Tutorial: https://mlflow.org/docs/latest/tutorials-and-examples/
- Streamlit Gallery: https://streamlit.io/gallery
- Docker Get Started: https://docs.docker.com/get-started/

### Community:
- Streamlit Forum: https://discuss.streamlit.io
- MLflow Slack: https://mlflow.org/community
- Stack Overflow: Tag with `streamlit`, `mlflow`, `docker`

---

## 🎯 Success Metrics

By the end of the session, participants should have:

**Technical:**
- [ ] Working Streamlit app running locally
- [ ] 5+ models trained and logged in MLflow
- [ ] Docker image built successfully
- [ ] App deployed to Streamlit Cloud
- [ ] Public URL they can share

**Conceptual:**
- [ ] Understand why experiment tracking matters
- [ ] Know how to compare ML models
- [ ] Grasp containerization benefits
- [ ] Understand basic CI/CD workflow
- [ ] Can adapt project for their own use

---

## ⚠️ Common Issues & Solutions

### "Module not found"
→ Activate virtual environment and reinstall: `pip install -r requirements.txt`

### "Port already in use"
→ Kill existing process or use different port

### "Model not found" error
→ Train at least one model first in the "Train Model" page

### MLflow UI empty
→ Refresh page after training, or check `mlruns/` directory exists

### Docker build fails
→ Ensure Docker is running, try `docker system prune -a`

**Full troubleshooting guide in README.md**


## 📝 Files You Need to Deploy

**Minimum files for deployment:**
```
mlops-iris/
├── streamlit_app.py       ✅ Required
├── requirements.txt       ✅ Required
└── .streamlit/
    └── config.toml        ⚠️ Optional (but recommended)
```

**All files for full experience:**
```
mlops-iris/
├── streamlit_app.py       ✅ Main app
├── requirements.txt       ✅ Dependencies
├── Dockerfile            ✅ For Docker
├── .gitignore            ✅ For Git
├── .streamlit/
│   └── config.toml       ✅ UI config
├── README.md             📖 For participants
├── TRAINER_GUIDE.md      📖 For trainer
└── CHEATSHEET.md         📖 Quick reference
```

---

## 💪 Why This Project Is Production-Ready

✅ **Error Handling** - Graceful failures with helpful messages  
✅ **Configuration** - Centralized settings  
✅ **Documentation** - Comprehensive guides  
✅ **Testing** - Clear validation steps  
✅ **Monitoring** - Health checks included  
✅ **Reproducibility** - Docker ensures consistency  
✅ **Version Control** - Git integration  
✅ **CI/CD** - Automatic deployment  
✅ **Scalability** - Easy to extend  
✅ **Best Practices** - Industry-standard patterns  

---

## 🎉 Final Notes

This project is designed to be:
- **Simple** enough for beginners
- **Realistic** enough for production
- **Complete** enough to be useful
- **Extensible** enough for research

**The goal:** Empower scientists to deploy their ML models confidently!

---

## 📧 Support

**During Training:**
- Ask your trainer
- Check CHEATSHEET.md
- Refer to README.md

**After Training:**
- GitHub Issues (if repo is public)
- Streamlit Community Forum
- MLflow Slack Channel

---

**Ready to deliver an awesome MLOps training session? Let's go! 🚀**

---

## 📄 License & Attribution

This is an educational project created for MLOps training.
Feel free to use, modify, and share with attribution.

**Created for:** MLOps Training Program  
**Version:** 1.0  
**Last Updated:** February 2026  

---

**Questions? Feedback? Improvements?**

We'd love to hear from you! Share your experience and help us improve this training material.

**Happy MLOps Learning! 🌸🚀**
