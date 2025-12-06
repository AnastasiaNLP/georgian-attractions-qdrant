# ☁️ Cloudinary Integration Guide

How images are stored and accessed in this project.

## Why Cloudinary?

- **Free Tier**: 25GB storage, 25GB bandwidth/month
- **CDN**: Fast delivery worldwide
- **Optimization**: Automatic image optimization
- **Reliability**: 99.9% uptime

## Image Upload Process

### 1. Dataset Images

Original images are in HuggingFace dataset as PIL Image objects.

### 2. Upload to Cloudinary
```bash
python3 tests/test_cloudinary_upload.py
```

**Process:**
- Loads dataset (1,715 records)
- Converts PIL Images to JPEG
- Uploads to Cloudinary
- Saves URL mapping to `image_urls.json`

**Time:** 30-60 minutes for 1,520 images

### 3. Update Qdrant
```bash
python3 tests/update_qdrant_images.py
```

Updates Qdrant records with Cloudinary URLs.

## Image URL Format
```
https://res.cloudinary.com/{cloud_name}/image/upload/{version}/georgian_attractions/{id}.jpg
```

Example:
```
https://res.cloudinary.com/dxwxd0gfb/image/upload/v1764980922/georgian_attractions/1.jpg
```

## Statistics

- **Total images in dataset**: 1,715
- **Uploaded to Cloudinary**: 1,520
- **Failed**: 2 (RGBA format not supported in JPEG)
- **Success rate**: 88.6%

## Failed Uploads

2 images failed due to RGBA format (PNG with transparency). JPEG doesn't support alpha channel.

**IDs**: 817, 818

These records still have metadata but `image_url` is `None`.

## Access Images

### In Python
```python
from PIL import Image
import requests
from io import BytesIO

# Get URL from Qdrant result
image_url = result.payload['image_url']

# Load image
response = requests.get(image_url)
img = Image.open(BytesIO(response.content))
img.show()
```

### In Web/HTML
```html
<img src="https://res.cloudinary.com/.../1.jpg" alt="Attraction">
```

## Cost Estimate

**Free Tier Limits:**
- Storage: 25GB
- Bandwidth: 25GB/month
- Transformations: 25,000/month

**Current Usage:**
- Storage: ~500MB-1GB (1,520 images)
- Well within free tier! ✅

## Alternatives

If you exceed free tier or want alternatives:

1. **AWS S3** - 5GB free (12 months)
2. **ImgBB** - Unlimited free (with API limits)
3. **GitHub** - Free but not recommended for large scale
4. **Self-hosted** - Use your own server

## Troubleshooting

### Upload fails with "Invalid cloud_name"

Check `CLOUDINARY_CLOUD_NAME` in `.env` - should be your actual cloud name, not "Root".

### 403 Forbidden

Check API credentials are correct.

### Slow uploads

Normal! Takes 1-3 seconds per image. Be patient.

## Image Optimization

Cloudinary automatically optimizes images:

- Format conversion (to WebP if browser supports)
- Compression
- Responsive sizing

Access optimized version:
```
https://res.cloudinary.com/.../image/upload/w_800,f_auto,q_auto/georgian_attractions/1.jpg
```

Parameters:
- `w_800`: width 800px
- `f_auto`: automatic format
- `q_auto`: automatic quality