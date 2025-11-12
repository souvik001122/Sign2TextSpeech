# Deploying the Sign Language Web App

## Local Testing

### 1. Install web dependencies
```bash
py -3.12 -m pip install -r requirements_web.txt
```

### 2. Run the Streamlit app
```bash
py -3.12 -m streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Deployment Options

### Option 1: Streamlit Community Cloud (Recommended - Free)

**Pros:** Free, easy, automatic deployment from GitHub

**Steps:**
1. Push your code to GitHub (already done: `souvik001122/Sign2TextSpeech`)
2. Go to https://share.streamlit.io/
3. Sign in with GitHub
4. Click "New app"
5. Select your repository: `souvik001122/Sign2TextSpeech`
6. Set main file path: `app.py`
7. Click "Deploy"

**Important Notes:**
- Upload model (`cnn8grps_rad1_model.h5`) as a GitHub Release asset
- Update `app.py` to download model from Release URL on first run
- Or use Git LFS for the model file

**Resource Limits:**
- Free tier: 1 GB RAM, 1 CPU
- Your model is ~13 MB (fits easily)
- May be slow for real-time video processing

### Option 2: Hugging Face Spaces (Free with GPU)

**Pros:** Free GPU, good for ML apps

**Steps:**
1. Create account at https://huggingface.co/
2. Create a new Space (select Streamlit)
3. Upload your files or connect GitHub repo
4. Add `requirements_web.txt` as requirements
5. Set `app.py` as entry point

**Limits:**
- Free tier includes basic GPU
- Better performance than Streamlit Cloud for ML

### Option 3: Render (Free tier available)

**Pros:** More resources than Streamlit Cloud

**Steps:**
1. Go to https://render.com/
2. Sign up and create new "Web Service"
3. Connect GitHub repo
4. Set build command: `pip install -r requirements_web.txt`
5. Set start command: `streamlit run app.py --server.port=$PORT`
6. Deploy

**Limits:**
- Free tier: 512 MB RAM
- Spins down after 15 minutes of inactivity

### Option 4: Heroku (Paid)

**Pros:** Reliable, scalable

**Cost:** ~$7/month minimum

**Not recommended for free deployment.**

### Option 5: AWS/GCP/Azure (Advanced)

**Pros:** Full control, scalable

**Cost:** Pay-as-you-go (can be expensive)

**Requirements:**
- EC2/Compute Engine/VM instance
- Configure security groups for HTTPS
- Install dependencies
- Use nginx + gunicorn for production

## Recommended Deployment Strategy

**For Demo/Portfolio:** Use Streamlit Community Cloud
- Easiest setup
- Free forever
- Good enough for demos
- URL: `https://your-app.streamlit.app`

**For Production:** Use Hugging Face Spaces or Render
- Better performance
- More reliable
- Still free or low-cost

## Configuration for Deployment

### Update app.py for cloud deployment

Add model download logic at the top of `app.py`:

```python
import urllib.request

MODEL_URL = "https://github.com/souvik001122/Sign2TextSpeech/releases/download/v1.0/cnn8grps_rad1_model.h5"

if not os.path.exists(MODEL_PATH):
    st.info("Downloading model file (first time only)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    st.success("Model downloaded!")
```

### Create streamlit config (optional)

Create `.streamlit/config.toml`:
```toml
[server]
headless = true
port = 8501

[browser]
gatherUsageStats = false
```

## Performance Notes

- **Webcam access:** Works in browser via WebRTC
- **Real-time processing:** May lag on free tiers
- **Model size:** 13 MB is acceptable for most platforms
- **Concurrent users:** Free tiers support 1-3 simultaneous users

## Security Notes

- HTTPS required for webcam access (cloud platforms provide this)
- Don't commit sensitive data or API keys
- Model file can be public (it's just weights)

## Troubleshooting

**"Cannot access webcam"**
- Ensure HTTPS (required by browsers)
- Check browser permissions

**"Model file not found"**
- Upload model as GitHub Release
- Add download logic to app.py

**"App is slow"**
- Reduce frame rate in webcam processing
- Use smaller model if possible
- Upgrade to paid tier for more resources

## Cost Comparison

| Platform | Free Tier | Performance | Best For |
|----------|-----------|-------------|----------|
| Streamlit Cloud | ✅ Unlimited | Fair | Demos |
| Hugging Face | ✅ With GPU | Good | ML Apps |
| Render | ✅ Limited | Fair | Side Projects |
| Heroku | ❌ $7/mo | Good | Production |

## Next Steps

1. Test locally first: `streamlit run app.py`
2. Upload model to GitHub Releases
3. Deploy to Streamlit Cloud
4. Share URL in README

## Support

For deployment issues:
- Streamlit: https://docs.streamlit.io/
- Hugging Face: https://huggingface.co/docs/hub/spaces
- Render: https://render.com/docs
