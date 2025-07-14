# main code를 다른 컴퓨터에서도 접근 가능하게 해준다.
## streamlit 서버를 어느 컴퓨터에서든지 열 수 있도록 해주는 코드이다.(단 같은 WIFI여야 함.)
> WIFI가 바뀌어 IP가 바뀌어도 접속 가능
>
> 하드 코딩으로 WIFI IP 설정했다가 겪은 반복된 오류 끝에 소프트 코딩으로 바꿈

```py
import os
import socket
import logging
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory, make_response
from werkzeug.utils import secure_filename
import stat

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleImageServer:
    def __init__(self, port=5000, upload_folder="./uploads"):
        self.app = Flask(__name__)
        self.port = port
        self.upload_folder = os.path.abspath(upload_folder)  # 절대 경로로 변경
        self.allowed_extensions = {"jpg", "jpeg", "png", "bmp", "gif"}
        self.max_file_size = 50 * 1024 * 1024  # 50MB

        # 업로드 폴더 생성 및 권한 설정
        self.setup_upload_folder()

        # Flask 설정
        self.app.config["MAX_CONTENT_LENGTH"] = self.max_file_size
        self.setup_routes()

    def setup_upload_folder(self):
        """업로드 폴더 생성 및 권한 설정"""
        try:
            os.makedirs(self.upload_folder, exist_ok=True)

            # 폴더 권한 설정 (읽기/쓰기/실행 권한)
            os.chmod(
                self.upload_folder,
                stat.S_IRWXU | stat.S_IRWXG | stat.S_IROTH | stat.S_IXOTH,
            )

            logger.info(f"업로드 폴더 설정 완료: {self.upload_folder}")

        except Exception as e:
            logger.error(f"업로드 폴더 생성 오류: {e}")
            # 임시 폴더로 대체
            import tempfile

            self.upload_folder = tempfile.mkdtemp()
            logger.info(f"임시 폴더 사용: {self.upload_folder}")

    def get_local_ip(self):
        """로컬 IP 주소 가져오기"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                return s.getsockname()[0]
        except:
            return "127.0.0.1"

    def allowed_file(self, filename):
        """허용된 파일 확장자 확인"""
        return (
            "." in filename
            and filename.rsplit(".", 1)[1].lower() in self.allowed_extensions
        )

    def setup_routes(self):
        """라우트 설정"""

        @self.app.route("/")
        def index():
            return jsonify(
                {
                    "status": "online",
                    "server": "Simple Image Server",
                    "upload_folder": self.upload_folder,
                    "endpoints": {
                        "upload": "/upload (POST)",
                        "files": "/files (GET)",
                        "download": "/files/<filename> (GET)",
                    },
                }
            )

        @self.app.route("/upload", methods=["POST"])
        def upload_file():
            try:
                # 업로드 폴더 존재 확인
                if not os.path.exists(self.upload_folder):
                    os.makedirs(self.upload_folder, exist_ok=True)

                if "file" not in request.files:
                    return jsonify({"error": "파일이 없습니다"}), 400

                file = request.files["file"]
                if file.filename == "":
                    return jsonify({"error": "파일이 선택되지 않았습니다"}), 400

                if not self.allowed_file(file.filename):
                    return jsonify({"error": "허용되지 않은 파일 형식"}), 400

                # 안전한 파일명 생성
                filename = secure_filename(file.filename)
                if not filename:  # secure_filename이 빈 문자열을 반환하는 경우
                    filename = f"upload_{int(datetime.now().timestamp())}.jpg"

                file_path = os.path.join(self.upload_folder, filename)

                # 중복 파일명 처리
                if os.path.exists(file_path):
                    name, ext = os.path.splitext(filename)
                    filename = f"{name}_{int(datetime.now().timestamp())}{ext}"
                    file_path = os.path.join(self.upload_folder, filename)

                # 파일 저장
                file.save(file_path)

                # 파일 권한 설정
                os.chmod(
                    file_path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH
                )

                file_size = os.path.getsize(file_path)

                logger.info(f"파일 업로드 성공: {filename} ({file_size} bytes)")

                return jsonify(
                    {
                        "success": True,
                        "filename": filename,
                        "size": f"{file_size/1024:.1f} KB",
                        "path": file_path,
                        "timestamp": datetime.now().isoformat(),
                    }
                )

            except Exception as e:
                logger.error(f"업로드 오류: {e}")
                return jsonify({"error": f"업로드 실패: {str(e)}"}), 500

        @self.app.route("/files")
        def list_files():
            try:
                # 업로드 폴더 존재 확인
                if not os.path.exists(self.upload_folder):
                    return jsonify(
                        {
                            "files": [],
                            "total": 0,
                            "message": "업로드 폴더가 존재하지 않습니다",
                        }
                    )

                files = []
                for filename in os.listdir(self.upload_folder):
                    file_path = os.path.join(self.upload_folder, filename)
                    if os.path.isfile(file_path):
                        try:
                            stat_info = os.stat(file_path)
                            files.append(
                                {
                                    "name": filename,
                                    "size": f"{stat_info.st_size/1024:.1f} KB",
                                    "modified": datetime.fromtimestamp(
                                        stat_info.st_mtime
                                    ).strftime("%Y-%m-%d %H:%M:%S"),
                                    "path": file_path,
                                }
                            )
                        except Exception as e:
                            logger.warning(f"파일 정보 읽기 실패 ({filename}): {e}")

                return jsonify(
                    {"files": files, "total": len(files), "folder": self.upload_folder}
                )

            except Exception as e:
                logger.error(f"파일 목록 조회 오류: {e}")
                return jsonify({"error": f"파일 목록 조회 실패: {str(e)}"}), 500

        @self.app.route("/files/<filename>")
        def download_file(filename):
            try:
                safe_filename = secure_filename(filename)
                file_path = os.path.join(self.upload_folder, safe_filename)

                if not os.path.exists(file_path):
                    return jsonify({"error": "파일을 찾을 수 없습니다"}), 404

                if not os.access(file_path, os.R_OK):
                    return jsonify({"error": "파일 접근 권한이 없습니다"}), 403

                response = make_response(
                    send_from_directory(self.upload_folder, safe_filename)
                )
                response.headers["Access-Control-Allow-Origin"] = "*"  # 🔥 중요
                return response

            except Exception as e:
                return jsonify({"error": f"파일 다운로드 실패: {str(e)}"}), 500

        @self.app.errorhandler(413)
        def too_large(e):
            return jsonify({"error": "파일 크기가 너무 큽니다 (최대 50MB)"}), 413

        @self.app.errorhandler(403)
        def forbidden(e):
            return jsonify({"error": "접근이 거부되었습니다"}), 403

        @self.app.errorhandler(404)
        def not_found(e):
            return jsonify({"error": "요청한 리소스를 찾을 수 없습니다"}), 404

    def check_permissions(self):
        """권한 확인"""
        try:
            # 업로드 폴더 권한 확인
            if os.path.exists(self.upload_folder):
                if not os.access(self.upload_folder, os.W_OK):
                    logger.warning(f"업로드 폴더 쓰기 권한 없음: {self.upload_folder}")
                if not os.access(self.upload_folder, os.R_OK):
                    logger.warning(f"업로드 폴더 읽기 권한 없음: {self.upload_folder}")

            # 테스트 파일 생성/삭제 시도
            test_file = os.path.join(self.upload_folder, "test_permission.txt")
            try:
                with open(test_file, "w") as f:
                    f.write("test")
                os.remove(test_file)
                logger.info("권한 테스트 통과")
            except Exception as e:
                logger.error(f"권한 테스트 실패: {e}")

        except Exception as e:
            logger.error(f"권한 확인 오류: {e}")

    def start(self):
        """서버 시작"""
        local_ip = self.get_local_ip()

        # 권한 확인
        self.check_permissions()

        print("=" * 60)
        print("🚀 Simple Image Server 시작")
        print(f"📡 주소: http://{local_ip}:{self.port}")
        print(f"📁 업로드 폴더: {self.upload_folder}")
        print(f"📏 최대 파일 크기: 50MB")
        print(f"📋 허용 확장자: {', '.join(self.allowed_extensions)}")
        print("-" * 60)
        print("📝 사용 방법:")
        print(f"   파일 업로드: POST http://{local_ip}:{self.port}/upload")
        print(f"   파일 목록: GET http://{local_ip}:{self.port}/files")
        print(f"   파일 다운로드: GET http://{local_ip}:{self.port}/files/<filename>")
        print("=" * 60)

        try:
            self.app.run(host="0.0.0.0", port=self.port, debug=False, threaded=True)
        except KeyboardInterrupt:
            print("\n서버가 종료되었습니다.")
        except Exception as e:
            logger.error(f"서버 시작 오류: {e}")


def main():
    server = SimpleImageServer(port=5000)
    server.start()


if __name__ == "__main__":
    main()
```
