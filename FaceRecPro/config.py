import os

# MySQL — override with environment variables in production (avoid committing secrets).
DB_CONFIG = {
    "host": os.environ.get("MYSQL_HOST", "localhost"),
    "user": os.environ.get("MYSQL_USER", "meghapranay"),
    "password": os.environ.get("MYSQL_PASSWORD", "1406"),
    "database": os.environ.get("MYSQL_DATABASE", "facerec_pro"),
}

# Recognition: lower distance = stricter (fewer false accepts, more false rejects).
MATCH_TOLERANCE = float(os.environ.get("FACE_MATCH_TOLERANCE", "0.52"))
# Frames must agree on the same identity before we show a name (reduces wrong-ID flashes).
IDENTITY_VOTE_WINDOW = int(os.environ.get("IDENTITY_VOTE_WINDOW", "9"))
IDENTITY_VOTE_MIN_RATIO = float(os.environ.get("IDENTITY_VOTE_MIN_RATIO", "0.67"))

# Enrollment: more jitters = stabler embedding, slower.
ENROLL_NUM_JITTERS = int(os.environ.get("ENROLL_NUM_JITTERS", "2"))

# LAB CLAHE on luminance — helps HOG + landmarks under backlight / low contrast.
USE_CLAHE = os.environ.get("USE_CLAHE", "true").lower() in ("1", "true", "yes")
CLAHE_CLIP_LIMIT = float(os.environ.get("CLAHE_CLIP_LIMIT", "2.5"))
CLAHE_GRID = (8, 8)

# Audit log: minimum seconds between DB rows for the same track (reduces spam).
HISTORY_LOG_INTERVAL_SEC = float(os.environ.get("HISTORY_LOG_INTERVAL", "3.5"))

# Face track: max center drift (pixels at full resolution) to consider same person.
TRACK_MATCH_MAX_DIST = float(os.environ.get("TRACK_MATCH_MAX_DIST", "120"))
