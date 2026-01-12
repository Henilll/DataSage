from flask import Flask, render_template, request, make_response, redirect, url_for, flash, jsonify
import requests
from playwright.sync_api import sync_playwright
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(
    __name__,
    static_folder="../static",
    static_url_path="/static"
)

app.secret_key = os.urandom(24)

BASE_URL = os.getenv("FASTAPI_BASE_URL")

FASTAPI_URL = f"{BASE_URL}/analyze"
CHAT_URL = f"{BASE_URL}/chat"
CLEANUP_URL = f"{BASE_URL}/cleanup"


app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max
ALLOWED_EXTENSIONS = {'csv'}


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def upload():
    """Home page with file upload"""
    if request.method == "POST":
        if 'file' not in request.files:
            flash('No file uploaded', 'error')
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)
        
        if not allowed_file(file.filename):
            flash('Only CSV files are allowed', 'error')
            return redirect(request.url)
        
        try:
            response = requests.post(
                FASTAPI_URL,
                files={"file": (file.filename, file.stream, file.mimetype)},
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Store data for PDF export
                app.config["DASHBOARD_DATA"] = data
                app.config["FILENAME"] = file.filename
                
                logger.info(f"Successfully analyzed: {file.filename}")
                return render_template("dashboard.html", data=data)
            else:
                error_msg = response.json().get('detail', 'Analysis failed')
                flash(f'Analysis error: {error_msg}', 'error')
                logger.error(f"Backend error: {error_msg}")
                return redirect(request.url)
        
        except requests.exceptions.ConnectionError:
            flash('Cannot connect to analysis server. Ensure FastAPI backend is running on port 8000.', 'error')
            logger.error("Connection error: FastAPI backend not reachable")
            return redirect(request.url)
        
        except requests.exceptions.Timeout:
            flash('Analysis timed out. Try with a smaller file.', 'error')
            logger.error("Timeout during file analysis")
            return redirect(request.url)
        
        except Exception as e:
            flash(f'Unexpected error: {str(e)}', 'error')
            logger.error(f"Unexpected error: {str(e)}")
            return redirect(request.url)
    
    return render_template("upload.html")


@app.route("/chat", methods=["POST"])
def chat():
    """
    Chatbot endpoint - proxies requests to FastAPI backend
    Allows the dashboard to communicate with the AI chatbot
    """
    try:
        data = request.get_json()
        
        if not data or 'session_id' not in data or 'question' not in data:
            return jsonify({"error": "Invalid request"}), 400
        
        # Forward request to FastAPI backend
        response = requests.post(
            CHAT_URL,
            json={
                "session_id": data['session_id'],
                "question": data['question']
            },
            timeout=30
        )
        
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            error_detail = response.json().get('detail', 'Chat error')
            logger.error(f"Chat error: {error_detail}")
            return jsonify({"error": error_detail}), response.status_code
    
    except requests.exceptions.Timeout:
        logger.error("Chat timeout")
        return jsonify({"error": "Request timed out"}), 504
    
    except requests.exceptions.ConnectionError:
        logger.error("Cannot connect to backend for chat")
        return jsonify({"error": "Cannot connect to AI service"}), 503
    
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/export")
def export_pdf():
    """Export dashboard as PDF"""
    if "DASHBOARD_DATA" not in app.config:
        flash('No dashboard data found. Upload a file first.', 'error')
        return redirect(url_for('upload'))
    
    try:
        html = render_template(
            "dashboard.html",
            data=app.config["DASHBOARD_DATA"]
        )
        
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            
            page.set_content(html, wait_until="networkidle")
            page.wait_for_timeout(2000)
            
            pdf = page.pdf(
                format="A4",
                print_background=True,
                margin={
                    "top": "20px",
                    "right": "20px",
                    "bottom": "20px",
                    "left": "20px"
                }
            )
            browser.close()
        
        response = make_response(pdf)
        response.headers["Content-Type"] = "application/pdf"
        
        filename = app.config.get("FILENAME", "analysis").replace('.csv', '')
        response.headers["Content-Disposition"] = (
            f"attachment; filename=DataSage_{filename}_Report.pdf"
        )
        
        logger.info("PDF export successful")
        return response
    
    except Exception as e:
        flash(f'PDF export failed: {str(e)}', 'error')
        logger.error(f"PDF export error: {str(e)}")
        return redirect(url_for('upload'))


@app.route("/cleanup")
def cleanup():
    """Cleanup old plot files"""
    try:
        response = requests.delete(CLEANUP_URL, timeout=10)
        if response.status_code == 200:
            result = response.json()
            flash(
                f'Cleanup complete: {result["files_deleted"]} files and '
                f'{result["sessions_cleaned"]} sessions removed',
                'success'
            )
        else:
            flash('Cleanup failed', 'error')
    except Exception as e:
        flash(f'Cleanup error: {str(e)}', 'error')
        logger.error(f"Cleanup error: {str(e)}")
    
    return redirect(url_for('upload'))


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large errors"""
    flash('File is too large. Maximum size is 50MB.', 'error')
    return redirect(url_for('upload'))


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    flash('Page not found', 'error')
    return redirect(url_for('upload'))


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal error: {error}")
    flash('An internal error occurred. Please try again.', 'error')
    return redirect(url_for('upload'))


if __name__ == "__main__":
    print("=" * 60)
    print("🚀 DataSage Frontend Server")
    print("=" * 60)
    print("📍 Access: http://127.0.0.1:5001")
    print("⚠️  Ensure FastAPI backend is running on port 8000")
    print("🤖 AI Chatbot: Enabled")
    print("=" * 60)
    
    app.run(debug=True, port=5001, host='0.0.0.0')
