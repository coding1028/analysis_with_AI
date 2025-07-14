# AI 불량률 측정기

## 시나리오는 다음과 같다.
1. 도장 공정 운영
- 전국의 중소 제조 기업들이 현장 환경 변화에 능동적으로 대응할 수 있도록 센서 기반 공정 데이터 수집 및 시각화 시스템 설계

2. 시스템 특징
- 이 시스템은 기업의 설비 규모나 인프라에 관계없이 적용 가능하도록 모듈화
- 실시간 공정 모니터링

최종 목표: 외부 환경 조건을 실시간으로 감지 및 기록
           현장 작업자가 공정 상태를 직관적으로 확인할 수 있도록 시각화

## 역할
이재찬(팀장)
가상 공정 구현, 로봇 데이터 관리 //
김민재(Me)
이미지 데이터 관리(with AI) //
박준우
이미지 데이터 게더링 및 전송 //
박민준
온/습도 센서 관리 / dash 가시화 //
이재민
뉴메릭 데이터 가시화 / dash 가시화 

## 공정 프로세스 및 결과물
<img width="1050" height="619" alt="Image" src="https://github.com/user-attachments/assets/a8eadb49-1dfb-41df-8645-ac6d7085f396" />

<img width="1662" height="447" alt="Image" src="https://github.com/user-attachments/assets/cda4867b-6566-4e7b-9ae7-6d9028241946" />


<img width="1745" height="1265" alt="Image" src="https://github.com/user-attachments/assets/c37f452c-df91-425e-a9eb-96529d3ee417" />

### 이미지 코드, tab5 5개를 분할하지 않고 합쳐놓았다. 
> main 기능: 전역 변수로 API 호출, gpt 모델을 이용해 불량률 검출(프롬프트를 통해 시나리오 가정한 걸 적용시킴)
> 
> 파인튜닝 프로세스를 통해 사용자가 AI의 불량률 판단을 재검토하여 성능 향상 가능


```py

import streamlit as st
from dotenv import load_dotenv
import os
from langchain_teddynote import logging
from langchain_openai import ChatOpenAI
from langchain_teddynote.models import MultiModal
import socket
from datetime import datetime
from datetime import timedelta
import tempfile
import shutil
import requests
import json
import time
import threading
from flask import Flask, request, jsonify, send_from_directory, make_response
from werkzeug.utils import secure_filename
import io
from PIL import Image
import base64
import queue
import glob
import pandas as pd
import csv
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
from openai import OpenAI
import zipfile
import sys
from flask_cors import CORS
import openai
from typing import List, Dict, Any


# API key 정보 로드
load_dotenv()

logging.langsmith("project 이미지 인식")

# 전역 변수 및 설정
UPLOAD_FOLDER = "received_images"
DATA_LOG_FOLDER = "analysis_logs"
FINETUNE_FOLDER = "finetuning_data"
FINETUNE_JOBS_FOLDER = "finetuning_jobs"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}

# 앱 이미지 업로드
image_queue = queue.Queue()
flask_app = Flask(__name__)
CORS(flask_app, resources={r"/files/*": {"origins": "*"}})
flask_app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# 폴더 생성
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATA_LOG_FOLDER, exist_ok=True)
os.makedirs(FINETUNE_FOLDER, exist_ok=True)
os.makedirs(FINETUNE_JOBS_FOLDER, exist_ok=True)

# CSV 로그 파일 경로
LOG_CSV_PATH = os.path.join(DATA_LOG_FOLDER, "analysis_logs.csv")  # 가시화 됨
TRAINING_DATA_PATH = os.path.join(FINETUNE_FOLDER, "training_data.jsonl")
VALIDATION_DATA_PATH = os.path.join(FINETUNE_FOLDER, "validation_data.jsonl")
FINETUNE_JOBS_PATH = os.path.join(FINETUNE_JOBS_FOLDER, "finetune_jobs.json")

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def initialize_csv_log():
    """CSV 로그 파일 초기화"""
    if not os.path.exists(LOG_CSV_PATH):
        headers = [
            "timestamp",
            "date",
            "time",
            "filename",
            "original_name",
            "file_size_mb",
            "image_width",
            "image_height",
            "model_used",
            "analysis_result",
            "judgment",
            "confidence_score",
            "processing_time_seconds",
            "defect_type",
            "notes",
            "user_feedback",
            "feedback_timestamp",
            "used_for_training",
        ]

        with open(LOG_CSV_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)


def initialize_finetune_jobs():
    """파인튜닝 작업 로그 초기화"""
    if not os.path.exists(FINETUNE_JOBS_PATH):
        with open(FINETUNE_JOBS_PATH, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)


def load_finetune_jobs():
    try:
        with open("finetune_jobs.json", "r", encoding="utf-8") as f:
            return json.load(f)  # 리스트 그대로 반환
    except Exception:
        return []


def save_finetune_jobs(jobs: list):
    """파인튜닝 작업 리스트를 JSON 파일로 저장"""
    try:
        with open(FINETUNE_JOBS_PATH, "w", encoding="utf-8") as f:
            json.dump(jobs, f, ensure_ascii=False, indent=2)
        print(f"✅ 작업 정보가 {FINETUNE_JOBS_PATH}에 저장되었습니다.")
    except Exception as e:
        print(f"❌ 작업 저장 중 오류 발생: {e}")


def add_finetune_job(job_data: Dict[str, Any]) -> bool:
    try:
        jobs = load_finetune_jobs()
        job_ids = {j.get("job_id") for j in jobs if "job_id" in j}
        if job_data.get("job_id") in job_ids:
            print(f"🔁 이미 존재하는 작업: {job_data['job_id']} 저장 안 함")
            return False
        jobs.append(job_data)
        saved = save_finetune_jobs(jobs)
        if saved:
            print(f"✅ 새 작업 저장 완료: {job_data['job_id']}")
        return saved
    except Exception as e:
        print(f"❌ 작업 추가 중 오류: {e}")
        return False


def extract_judgment_info(analysis_text):
    """분석 결과에서 판정 정보 추출"""
    analysis_lower = analysis_text.lower()

    # 판정 결과 추출
    if "불량" in analysis_lower:
        judgment = "불량"
    elif "정품" in analysis_lower or "정상" in analysis_lower:
        judgment = "정품"
    else:
        judgment = "미분류"

    # 신뢰도 추출 (% 기호가 있는 숫자 찾기)
    import re

    confidence_match = re.search(r"(\d+(?:\.\d+)?)\s*%", analysis_text)
    confidence_score = float(confidence_match.group(1)) if confidence_match else None

    # 불량 유형 추출 (간단한 키워드 기반)
    defect_keywords = {
        "색상": ["색상", "컬러", "빨간색", "파란색"],
        "형태": ["형태", "모양", "변형"],
        "표면": ["표면", "스크래치", "긁힘"],
        "크기": ["크기", "사이즈", "치수"],
    }

    defect_type = "기타"
    for defect, keywords in defect_keywords.items():
        if any(keyword in analysis_lower for keyword in keywords):
            defect_type = defect
            break

    return judgment, confidence_score, defect_type


def get_image_info(image_path):
    """이미지 파일 정보 추출"""
    try:
        # 파일 크기 (MB)
        file_size_mb = round(os.path.getsize(image_path) / (1024 * 1024), 2)

        # 이미지 크기
        with Image.open(image_path) as img:
            width, height = img.size

        return file_size_mb, width, height
    except Exception as e:
        st.error(f"이미지 정보 추출 오류: {e}")
        return 0, 0, 0


def log_analysis_result(
    image_path, original_name, analysis_result, model_used, processing_time
):
    """분석 결과를 CSV에 기록"""
    try:
        now = datetime.now()

        # 이미지 정보 추출
        file_size_mb, width, height = get_image_info(image_path)

        # 판정 정보 추출
        judgment, confidence_score, defect_type = extract_judgment_info(analysis_result)

        # 로그 데이터 준비
        log_data = [
            now.isoformat(),  # timestamp
            now.strftime("%Y-%m-%d"),  # date
            now.strftime("%H:%M:%S"),  # time
            os.path.basename(image_path),  # filename
            original_name,  # original_name
            file_size_mb,  # file_size_mb
            width,  # image_width
            height,  # image_height
            model_used,  # model_used
            analysis_result.replace("\n", " "),  # analysis_result (개행 제거)
            judgment,  # judgment
            confidence_score,  # confidence_score
            round(processing_time, 2),  # processing_time_seconds
            defect_type,  # defect_type
            "",  # notes (추후 수동 입력 가능)
            "",  # user_feedback
            "",  # feedback_timestamp
            "No",  # used_for_training
        ]

        # CSV에 추가
        with open(LOG_CSV_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(log_data)

        return True
    except Exception as e:
        st.error(f"로그 기록 오류: {e}")
        return False


if "finetune_jobs_json" not in st.session_state:
    st.session_state.finetune_jobs_json = load_finetune_jobs()


def update_user_feedback(index, feedback, correct_judgment):
    """사용자 피드백 업데이트"""
    try:
        df = pd.read_csv(LOG_CSV_PATH, encoding="utf-8")

        if index < len(df):
            df.loc[index, "user_feedback"] = correct_judgment
            df.loc[index, "feedback_timestamp"] = datetime.now().isoformat()
            df.loc[index, "notes"] = feedback

            df.to_csv(LOG_CSV_PATH, index=False)
            return True
        return False
    except Exception as e:
        st.error(f"피드백 업데이트 오류: {e}")
        return False


def prepare_training_data():
    """훈련 데이터 준비"""
    try:
        df = pd.read_csv(LOG_CSV_PATH, encoding="utf-8")

        # 피드백이 있는 데이터만 필터링
        feedback_data = df[df["user_feedback"].notna() & (df["user_feedback"] != "")]

        if len(feedback_data) == 0:
            return 0, "피드백 데이터가 없습니다."

        training_samples = []

        for _, row in feedback_data.iterrows():
            # 시스템 메시지
            system_message = {
                "role": "system",
                "content": st.session_state.get(
                    "system_prompt",
                    "당신은 이미지를 분석하여 도장 불량률을 판단하는 전문 AI 어시스턴트입니다.",
                ),
            }

            # 사용자 메시지 (이미지 분석 요청)
            user_message = {
                "role": "user",
                "content": "이 이미지를 분석하여 불량/정품을 판단해주세요.",
            }

            # 어시스턴트 메시지 (올바른 답변)
            correct_judgment = row["user_feedback"]
            assistant_content = f"분석 결과: {correct_judgment}"

            if correct_judgment == "불량":
                assistant_content += "\n이 제품은 불량품으로 판정됩니다."
            elif correct_judgment == "정품":
                assistant_content += "\n이 제품은 정품으로 판정됩니다."

            assistant_message = {"role": "assistant", "content": assistant_content}

            # 훈련 샘플 생성
            training_sample = {
                "messages": [system_message, user_message, assistant_message]
            }

            training_samples.append(training_sample)

        # 훈련/검증 데이터 분할 (80:20)
        split_idx = int(len(training_samples) * 0.8)
        training_data = training_samples[:split_idx]
        validation_data = training_samples[split_idx:]

        # JSONL 파일로 저장
        with open(TRAINING_DATA_PATH, "w", encoding="utf-8") as f:
            for sample in training_data:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        if validation_data:
            with open(VALIDATION_DATA_PATH, "w", encoding="utf-8") as f:
                for sample in validation_data:
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        # 사용된 데이터 마킹
        used_indices = feedback_data.index.tolist()
        df.loc[used_indices, "used_for_training"] = "Yes"
        df.to_csv(LOG_CSV_PATH, index=False)

        return (
            len(training_samples),
            f"훈련 데이터 {len(training_data)}개, 검증 데이터 {len(validation_data)}개 준비 완료",
        )

    except Exception as e:
        return 0, f"훈련 데이터 준비 오류: {e}"


def cancel_job(job_id: str):
    """파인튜닝 작업을 취소합니다."""
    try:
        # 환경변수에서 API 키 가져오기
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise Exception(
                "❌ OPENAI_API_KEY 환경변수가 설정되지 않았습니다. .env 파일을 확인해주세요."
            )

        # OpenAI 클라이언트 생성
        client = openai.OpenAI(api_key=api_key)

        print("작업을 취소하는 중...")

        # 파인튜닝 작업 취소
        response = client.fine_tuning.jobs.cancel(job_id)

        if response.status == "cancelled":
            print("✅ 작업이 취소되었습니다.")

            # 로컬 목록 업데이트 (load_finetune_jobs, save_finetune_jobs 함수가 있다고 가정)
            try:
                jobs = load_finetune_jobs()
                for job in jobs:
                    if job.get("job_id") == job_id:
                        job["status"] = "cancelled"
                        job["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        break
                save_finetune_jobs(jobs)
                print("로컬 작업 목록이 업데이트되었습니다.")
            except Exception as local_update_error:
                print(f"⚠️ 로컬 목록 업데이트 중 오류: {local_update_error}")

            return True
        else:
            print(f"작업 상태: {response.status}")
            return False

    except openai.APIError as e:
        if "not found" in str(e).lower():
            raise Exception(f"작업 ID '{job_id}'를 찾을 수 없습니다.")
        elif "insufficient_quota" in str(e):
            raise Exception("API 사용량이 부족합니다. 결제 정보를 확인해주세요.")
        else:
            raise Exception(f"OpenAI API 오류: {str(e)}")
    except Exception as e:
        raise Exception(f"작업 취소 중 오류 발생: {str(e)}")


def check_job_status(job_id: str):
    """특정 작업의 상태를 확인합니다."""
    try:
        # 환경변수에서 API 키 가져오기
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise Exception(
                "❌ OPENAI_API_KEY 환경변수가 설정되지 않았습니다. .env 파일을 확인해주세요."
            )

        # OpenAI 클라이언트 생성
        client = openai.OpenAI(api_key=api_key)

        print("작업 상태를 확인하는 중...")

        # 파인튜닝 작업 상태 조회
        response = client.fine_tuning.jobs.retrieve(job_id)

        # 상태 정보 표시
        status_emoji = {
            "running": "🟡",
            "succeeded": "🟢",
            "failed": "🔴",
            "cancelled": "⚫",
        }.get(response.status, "⚪")

        print(f"{status_emoji} 현재 상태: {response.status}")

        # 상태별 추가 정보 표시
        if response.status == "succeeded":
            print(f"🎉 파인튜닝이 완료되었습니다!")
            print(f"모델: {response.fine_tuned_model}")
        elif response.status == "failed":
            error_msg = response.error.message if response.error else "Unknown error"
            print(f"❌ 파인튜닝이 실패했습니다: {error_msg}")
        elif response.status == "running":
            print("⏳ 파인튜닝이 진행 중입니다...")
        elif response.status == "cancelled":
            print("⚫ 작업이 취소되었습니다.")

        # 기본 작업 정보 출력
        print(f"\n작업 세부 정보:")
        print(f"- 작업 ID: {response.id}")
        print(f"- 모델: {response.model}")
        print(
            f"- 생성 시간: {datetime.fromtimestamp(response.created_at) if response.created_at else 'N/A'}"
        )

        if response.finished_at:
            print(f"- 완료 시간: {datetime.fromtimestamp(response.finished_at)}")

        if response.trained_tokens:
            print(f"- 훈련된 토큰 수: {response.trained_tokens:,}")

        # 로컬 목록 업데이트
        try:
            jobs = load_finetune_jobs()
            for job in jobs:
                if job.get("job_id") == job_id:
                    job["status"] = response.status
                    job["finished_at"] = (
                        datetime.fromtimestamp(response.finished_at).strftime(
                            "%Y-%m-%d %H:%M:%S"
                        )
                        if response.finished_at
                        else None
                    )
                    job["fine_tuned_model"] = response.fine_tuned_model
                    job["trained_tokens"] = response.trained_tokens
                    job["error"] = response.error.to_dict() if response.error else None
                    job["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    break

            save_finetune_jobs(jobs)
            print("로컬 작업 목록이 업데이트되었습니다.")
        except Exception as local_update_error:
            print(f"⚠️ 로컬 목록 업데이트 중 오류: {local_update_error}")

        # 상태 정보 딕셔너리로 반환
        return {
            "id": response.id,
            "status": response.status,
            "model": response.model,
            "fine_tuned_model": response.fine_tuned_model,
            "created_at": response.created_at,
            "finished_at": response.finished_at,
            "trained_tokens": response.trained_tokens,
            "error": response.error.to_dict() if response.error else None,
        }

    except openai.APIError as e:
        if "not found" in str(e).lower():
            raise Exception(f"작업 ID '{job_id}'를 찾을 수 없습니다.")
        elif "insufficient_quota" in str(e):
            raise Exception("API 사용량이 부족합니다. 결제 정보를 확인해주세요.")
        else:
            raise Exception(f"OpenAI API 오류: {str(e)}")
    except Exception as e:
        raise Exception(f"상태 확인 중 오류 발생: {str(e)}")


def upload_training_file(file_path):
    """OpenAI에 훈련 파일 업로드"""
    try:
        with open(file_path, "rb") as f:
            response = client.files.create(file=f, purpose="fine-tune")
        return response.id, None
    except Exception as e:
        return None, str(e)


def create_finetune_job(
    training_file_id, validation_file_id=None, model="gpt-3.5-turbo-1106"
):
    """파인튜닝 작업 생성"""
    try:
        hyperparameters = {
            "n_epochs": st.session_state.get("finetune_epochs", 3),
            "batch_size": st.session_state.get("finetune_batch_size", 1),
            "learning_rate_multiplier": st.session_state.get(
                "finetune_learning_rate", 2.0
            ),
        }

        job_params = {
            "training_file": training_file_id,
            "model": model,
            "hyperparameters": hyperparameters,
        }

        if validation_file_id:
            job_params["validation_file"] = validation_file_id

        response = client.fine_tuning.jobs.create(**job_params)

        return response.id, None
    except Exception as e:
        return None, str(e)


def get_finetune_job_status(job_id):
    """파인튜닝 작업 상태 확인"""
    try:
        response = client.fine_tuning.jobs.retrieve(job_id)
        return response, None
    except Exception as e:
        return None, str(e)


def list_finetune_jobs():
    try:
        response = client.fine_tuning.jobs.list()
        return response.data, None
    except Exception as e:
        return None, str(e)


def load_analysis_logs():
    """분석 로그 CSV를 읽어 DataFrame으로 반환"""

    columns = [
        "timestamp",  # ex: 2025-06-14T14:33:40.701977
        "date",  # ex: 2025-06-14
        "time",  # ex: 14:33:40
        "full_filename",  # ex: 20250614_143326_964332_camera_capture...
        "filename",  # ex: camera_capture_20250614_143326_382309.jpg
        "confidence_score",  # ex: 0.18
        "width",  # ex: 1280
        "height",  # ex: 720
        "model_used",  # ex: gpt-4o
        "description",  # ex: 전체적인 내용...
        "judgment",  # ex: 불량
        "score",  # ex: 95.0
        "processing_time_seconds",  # ex: 3.98
        "defect_type",  # ex: 색상
        "Unnamed: 14",  # 불필요, 제거 가능
        "Unnamed: 15",
        "Unnamed: 16",
        "user_feedback",  # ex: No
    ]

    if not os.path.exists(LOG_CSV_PATH):
        return pd.DataFrame(columns=columns)

    try:
        # header=None, names=columns : CSV에 헤더가 없거나 컬럼명을 덮어쓸 때 사용
        df = pd.read_csv(LOG_CSV_PATH, encoding="utf-8")
        return df
    except Exception as e:
        st.error(f"로그 파일 로드 중 오류 발생: {e}")
        return pd.DataFrame(columns=columns)


def allowed_file(filename):
    """허용된 파일 확장자 확인"""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# Flask API 엔드포인트들
@flask_app.route("/upload", methods=["POST"])
def upload_image():
    """이미지 업로드 API"""
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        if file and allowed_file(file.filename):
            # 안전한 파일명 생성
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{timestamp}_{secure_filename(file.filename)}"
            filepath = os.path.join(flask_app.config["UPLOAD_FOLDER"], filename)

            # 파일 저장
            file.save(filepath)

            # 큐에 새 이미지 정보 추가
            image_queue.put(
                {
                    "filename": filename,
                    "filepath": filepath,
                    "timestamp": timestamp,
                    "original_name": file.filename,
                    "upload_time": datetime.now().isoformat(),
                }
            )

            return (
                jsonify(
                    {
                        "message": "Image uploaded successfully",
                        "filename": filename,
                        "timestamp": timestamp,
                    }
                ),
                200,
            )
        else:
            return jsonify({"error": "Invalid file type"}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@flask_app.route("/images/<path:filename>")
def serve_image(filename):
    try:
        response = make_response(
            send_from_directory(flask_app.config["UPLOAD_FOLDER"], filename)
        )
        response.headers["Access-Control-Allow-Origin"] = "*"
        return response
    except Exception as e:
        return jsonify({"error": f"Cannot serve image: {str(e)}"}), 404


@flask_app.route("/upload_base64", methods=["POST"])
def upload_base64_image():
    """Base64 인코딩된 이미지 업로드 API"""
    try:
        data = request.get_json()
        if "image" not in data:
            return jsonify({"error": "No image data provided"}), 400

        # Base64 디코딩
        image_data = data["image"]
        if "data:image" in image_data:
            # data URL 형식인 경우 헤더 제거
            image_data = image_data.split(",")[1]

        # 이미지 디코딩 및 저장
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))

        # 파일명 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{timestamp}_camera_capture.jpg"
        filepath = os.path.join(flask_app.config["UPLOAD_FOLDER"], filename)

        # 이미지 저장
        image.save(filepath, "JPEG")

        # 큐에 새 이미지 정보 추가
        image_queue.put(
            {
                "filename": filename,
                "filepath": filepath,
                "timestamp": timestamp,
                "original_name": "camera_capture.jpg",
                "upload_time": datetime.now().isoformat(),
            }
        )

        return (
            jsonify(
                {
                    "message": "Image uploaded successfully",
                    "filename": filename,
                    "timestamp": timestamp,
                }
            ),
            200,
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@flask_app.route("/status", methods=["GET"])
def get_status():
    """서버 상태 확인 API"""
    return jsonify(
        {
            "status": "running",
            "timestamp": datetime.now().isoformat(),
            "queue_size": image_queue.qsize(),
        }
    )


@flask_app.route("/health", methods=["GET"])
def health_check():
    """헬스체크 API"""
    return jsonify(
        {
            "status": "healthy",
            "server": "AI Image Analyzer",
            "version": "3.0",
            "timestamp": datetime.now().isoformat(),
        }
    )


def run_flask_server():
    """Flask 서버 실행"""
    flask_app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)


# 서버 정보 표시 함수
def get_server_info():
    """현재 서버 정보 가져오기"""
    hostname = socket.gethostname()
    try:
        # 더 정확한 로컬 IP 가져오기
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except:
        local_ip = socket.gethostbyname(hostname)
    return hostname, local_ip


def get_received_images():
    """받은 이미지 목록 가져오기"""
    image_files = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        image_files.extend(glob.glob(os.path.join(UPLOAD_FOLDER, ext)))

    # 최신 순으로 정렬
    image_files.sort(key=os.path.getctime, reverse=True)
    return image_files


def generate_image_analysis(image_path, system_prompt, model_name="gpt-4o"):
    """이미지 분석 수행"""
    start_time = time.time()

    try:
        llm = ChatOpenAI(
            temperature=0,
            model_name=model_name,
        )

        # 기본 분석 프롬프트
        user_prompt = """
        이 이미지를 자세히 분석해 주세요. 다음 항목들을 포함해서 설명해 주세요:
        
        1. **전체적인 내용**: 이미지에서 보이는 물체
        2. **객체 및 요소**: 이미지에 물체를 감싸고 있는 종이의 색깔
        3. **품질 판단**: 불량/정품 판정 결과
        4. **신뢰도**: 판정에 대한 확신 정도 (%)

        """

        multimodal_llm = MultiModal(
            llm, system_prompt=system_prompt, user_prompt=user_prompt
        )

        # 응답 받기 및 내용 추출
        response = multimodal_llm.invoke(image_path)

        processing_time = time.time() - start_time

        # 응답 타입에 따른 처리
        if hasattr(response, "content"):
            # AIMessage 객체인 경우
            result = response.content
        elif isinstance(response, str):
            # 이미 문자열인 경우
            result = response
        else:
            # 기타 경우 - 문자열로 변환 시도
            result = str(response)

        return result, processing_time

    except Exception as e:
        processing_time = time.time() - start_time

        # 더 자세한 에러 정보 제공
        error_msg = f"❌ 이미지 분석 중 오류가 발생했습니다.\n"
        error_msg += f"오류 유형: {type(e).__name__}\n"
        error_msg += f"오류 내용: {str(e)}\n"
        error_msg += f"이미지 경로: {image_path}\n"
        error_msg += f"모델: {model_name}"

        # 로그에도 기록
        st.error(f"Image analysis error: {str(e)}")

        return error_msg, processing_time


def auto_analyze_new_images():
    """새로운 이미지 자동 분석"""
    if "analyzed_images" not in st.session_state:
        st.session_state.analyzed_images = set()

    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = {}

    received_images = get_received_images()

    for image_path in received_images[:5]:  # 최근 5개만 자동 분석
        if image_path not in st.session_state.analyzed_images:
            # 자동 분석 수행
            system_prompt = st.session_state.get(
                "system_prompt",
                "당신은 이미지를 분석하여 도장 불량률을 판단하는 전문 AI 어시스턴트입니다. 이미지의 물체가 빨간색 종이로 감싸져 있으면 불량으로, 파란색으로 감싸져 있다면 정품으로 판단해주세요.",
            )

            model_name = st.session_state.get("selected_model", "gpt-4o")

            result, processing_time = generate_image_analysis(
                image_path, system_prompt, model_name
            )

            # 결과 저장
            analysis_info = {
                "result": result,
                "timestamp": datetime.now(),
                "model": model_name,
                "processing_time": processing_time,
            }

            st.session_state.analysis_results[image_path] = analysis_info
            st.session_state.analyzed_images.add(image_path)

            # CSV 로그에 기록
            original_name = (
                os.path.basename(image_path).split("_", 3)[-1]
                if "_" in os.path.basename(image_path)
                else os.path.basename(image_path)
            )
            log_analysis_result(
                image_path, original_name, result, model_name, processing_time
            )


# CSV 로그 파일 초기화
initialize_csv_log()
initialize_finetune_jobs()

# 페이지 설정
st.set_page_config(
    page_title="AI 이미지 분석기 with 파인튜닝 ",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 탭 구성
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📡 실시간 모니터링", "📊 데이터 분석", "🔧 파인튜닝", "📝 피드백 관리", "⚙️ 설정"]
)


def delete_all_images():
    """모든 저장된 이미지 파일 삭제"""
    try:
        deleted_count = 0

        # temp_captures 폴더의 모든 이미지 파일 삭제
        if os.path.exists("temp_captures"):
            image_files = glob.glob("temp_captures/*")
            for file_path in image_files:
                if os.path.isfile(file_path) and file_path.lower().endswith(
                    (".jpg", ".jpeg", ".png", ".bmp", ".gif")
                ):
                    os.remove(file_path)
                    deleted_count += 1

        return deleted_count, None
    except Exception as e:
        return 0, str(e)


def refresh_received_images():
    """파일 시스템과 세션 상태 동기화"""
    try:
        folder = UPLOAD_FOLDER  # "received_images"가 될 것
        if os.path.exists(folder):
            actual_files = []
            for file_path in glob.glob(os.path.join(folder, "*")):
                if os.path.isfile(file_path) and file_path.lower().endswith(
                    (".jpg", ".jpeg", ".png", ".bmp", ".gif")
                ):
                    actual_files.append(file_path)

            # 세션 상태 업데이트
            if "received_images" in st.session_state:
                st.session_state.received_images = [
                    img
                    for img in st.session_state.received_images
                    if img in actual_files
                ]

            if "analysis_results" in st.session_state:
                existing_results = {
                    img_path: result
                    for img_path, result in st.session_state.analysis_results.items()
                    if img_path in actual_files
                }
                st.session_state.analysis_results = existing_results

            return actual_files
        else:
            return []
    except Exception as e:
        st.error(f"파일 동기화 오류: {str(e)}")
        return []


def delete_selected_images(image_paths):
    """선택된 이미지 파일들 삭제"""
    try:
        deleted_count = 0
        failed_files = []

        for img_path in image_paths:
            try:
                if os.path.exists(img_path):
                    os.remove(img_path)
                    deleted_count += 1
            except Exception as e:
                failed_files.append(f"{img_path}: {str(e)}")

        return deleted_count, failed_files
    except Exception as e:
        return 0, [str(e)]


def clear_all_analysis_logs():
    """모든 분석 로그를 삭제하는 함수 (빈 CSV로 초기화)"""
    try:
        empty_df = pd.DataFrame(
            columns=[
                "timestamp",
                "filename",
                "judgment",
                "confidence_score",
                "processing_time_seconds",
                "model_used",
                "user_feedback",
                "date",
                "defect_type",
            ]
        )
        empty_df.to_csv(LOG_CSV_PATH, index=False)
        return True, None
    except Exception as e:
        return False, str(e)


def clear_temp_folder():
    """temp_captures 폴더 전체 삭제 후 재생성"""
    try:
        if os.path.exists("temp_captures"):
            shutil.rmtree("temp_captures")

        os.makedirs("temp_captures", exist_ok=True)
        return True, None
    except Exception as e:
        return False, str(e)


# 기존 코드에 추가할 부분
with tab1:
    # 메인 제목
    st.title("AI 이미지 분석")
    st.markdown("### 이미지 데이터 가시화 및 파인튜닝")

    # 서버 정보 표시
    hostname, local_ip = get_server_info()
    st.info(f"🌐 서버 정보 - 호스트: {hostname} | 접속 IP: {local_ip}")

    # 이미지 관리 버튼들 (상단에 배치)
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        if st.button("🗑️ 모든 이미지 삭제", type="secondary"):
            deleted_count, error = delete_all_images()
            if error:
                st.error(f"❌ 삭제 실패: {error}")
            else:
                # 세션 상태 완전 초기화
                st.session_state.received_images = []
                st.session_state.analysis_results = {}
                st.session_state.analyzed_images = set()

                # 큐도 비우기
                while not image_queue.empty():
                    try:
                        image_queue.get_nowait()
                    except queue.Empty:
                        break

                st.success(f"✅ {deleted_count}개 이미지가 삭제되었습니다!")
                st.rerun()

    with col2:
        if st.button("🔄 폴더 초기화", type="secondary"):
            success, error = clear_temp_folder()
            if error:
                st.error(f"❌ 초기화 실패: {error}")
            else:
                # 세션 상태 완전 초기화
                st.session_state.received_images = []
                st.session_state.analysis_results = {}
                st.session_state.analyzed_images = set()

                # 큐도 비우기
                while not image_queue.empty():
                    try:
                        image_queue.get_nowait()
                    except queue.Empty:
                        break

                st.success("✅ 이미지 폴더가 초기화되었습니다!")
                st.rerun()

    # 파일 시스템과 세션 상태 동기화
    actual_files = refresh_received_images()

    # Flask 서버 시작 (세션 상태로 중복 실행 방지)
    if "flask_started" not in st.session_state:
        try:
            flask_thread = threading.Thread(target=run_flask_server, daemon=True)
            flask_thread.start()
            st.session_state.flask_started = True
            st.success("🚀 이미지 수신 서버가 시작되었습니다!")
        except Exception as e:
            st.error(f"❌ 서버 시작 실패: {str(e)}")

    # 새로운 이미지 체크 및 처리
    # 세션 상태 초기화
    if "received_images" not in st.session_state:
        st.session_state.received_images = []

    # 이미지 큐 수신
    new_images = []
    while True:
        try:
            new_image = image_queue.get_nowait()
            new_images.append(new_image)
            st.session_state.received_images.insert(0, new_image)
        except queue.Empty:
            break

    # 자동 분석 설정 (사이드바에서 가져오기)
    auto_analysis = st.session_state.get("auto_analysis", True)

    # 자동 분석 수행
    if auto_analysis:
        auto_analyze_new_images()

    # 새 이미지가 도착했을 때 알림
    if new_images:
        st.success(f"🆕 새로운 이미지 {len(new_images)}개가 도착했습니다!")
        for img_info in new_images:
            st.info(f"📷 {img_info['original_name']} ({img_info['timestamp']})")

    # 받은 이미지 목록 표시 (파일 시스템과 동기화된 목록 사용)
    received_images = actual_files  # get_received_images() 대신 actual_files 사용

    if received_images:
        st.write(f"**총 {len(received_images)}개의 이미지를 받았습니다**")

        # 개별 삭제를 위한 선택 옵션
        st.markdown("#### 개별 이미지 삭제")
        selected_for_deletion = []

        # 최근 이미지들을 그리드로 표시
        cols = st.columns(3)

        for idx, img_path in enumerate(received_images[:9]):  # 최근 9개만 표시
            with cols[idx % 3]:
                # 삭제 선택 체크박스
                delete_selected = st.checkbox(
                    f"삭제 선택",
                    key=f"delete_check_{idx}",
                    help="이 이미지를 삭제하려면 체크하세요",
                )

                if delete_selected:
                    selected_for_deletion.append(img_path)

                # 이미지 표시
                st.image(
                    img_path,
                    caption=f"{os.path.basename(img_path)}",
                    use_column_width=True,
                )

                # 분석 결과가 있으면 표시
                if img_path in st.session_state.get("analysis_results", {}):
                    analysis_info = st.session_state.analysis_results[img_path]

                    # 불량/정품 판정 결과 하이라이트
                    result_text = analysis_info["result"]
                    if "불량" in result_text:
                        st.error("🚨 불량 판정")
                    elif "정품" in result_text or "정상" in result_text:
                        st.success("✅ 정품 판정")
                    else:
                        st.info("🔍 분석 완료")

                    # 상세 결과 보기
                    with st.expander(f"상세 결과 보기", expanded=False):
                        st.write(
                            f"**분석 시간:** {analysis_info['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}"
                        )
                        st.write(
                            f"**처리 시간:** {analysis_info['processing_time']:.2f}초"
                        )
                        st.write("**분석 결과:**")
                        st.write(result_text)

                # 수동 분석 버튼
                if st.button(f"🔍 수동 분석", key=f"analyze_{idx}"):
                    system_prompt = st.session_state.get(
                        "system_prompt",
                        "당신은 이미지를 분석하여 도장 불량률을 판단하는 전문 AI 어시스턴트입니다.",
                    )
                    model_name = st.session_state.get("selected_model", "gpt-4o")

                    with st.spinner("분석 중..."):
                        result, processing_time = generate_image_analysis(
                            img_path, system_prompt, model_name
                        )

                        # 결과 저장
                        analysis_info = {
                            "result": result,
                            "timestamp": datetime.now(),
                            "model": model_name,
                            "processing_time": processing_time,
                        }

                        if "analysis_results" not in st.session_state:
                            st.session_state.analysis_results = {}

                        st.session_state.analysis_results[img_path] = analysis_info
                        st.session_state.analyzed_images.add(img_path)

                        # CSV 로그에 기록
                        original_name = (
                            os.path.basename(img_path).split("_", 3)[-1]
                            if "_" in os.path.basename(img_path)
                            else os.path.basename(img_path)
                        )
                        log_analysis_result(
                            img_path, original_name, result, model_name, processing_time
                        )

                        st.rerun()

        # 선택된 이미지들 삭제 버튼
        if selected_for_deletion:
            st.markdown("---")
            col_del1, col_del2 = st.columns([1, 3])

            with col_del1:
                if st.button(
                    f"🗑️ 선택된 {len(selected_for_deletion)}개 삭제", type="primary"
                ):
                    deleted_count, failed_files = delete_selected_images(
                        selected_for_deletion
                    )

                    if failed_files:
                        st.error(f"❌ 일부 파일 삭제 실패: {failed_files}")

                    if deleted_count > 0:
                        # 파일 시스템과 세션 상태 다시 동기화
                        refresh_received_images()

                        st.success(f"✅ {deleted_count}개 이미지가 삭제되었습니다!")
                        st.rerun()

    else:
        st.info(
            "📷 아직 받은 이미지가 없습니다. 카메라나 API를 통해 이미지를 업로드해 주세요."
        )

        # API 사용 예시
        st.markdown("### 📡 API 사용 방법")
        st.code(
            f"""
http:// 웹 사이트에 입력
        """,
            language="bash",
        )


def delete_all_analysis_data():
    """전체 삭제를 위한 래퍼 함수"""
    success, error = clear_all_analysis_logs()
    return success, error


def delete_selected_analysis_data(timestamps):
    """선택된 timestamp 리스트에 해당하는 로그 삭제 함수"""
    try:
        df = load_analysis_logs()
        if df.empty:
            return 0, None
        initial_count = len(df)
        df_filtered = df[~df["timestamp"].isin(timestamps)]
        df_filtered.to_csv(LOG_CSV_PATH, index=False)
        deleted_count = initial_count - len(df_filtered)
        return deleted_count, None
    except Exception as e:
        return 0, str(e)


def refresh_analysis_data():
    """분석 로그를 최신으로 다시 불러오는 함수 (탭2에서 호출용)"""
    return load_analysis_logs()


with tab2:
    st.header("데이터 분석 대시보드")

    # 분석 로그 로드 (새로고침 함수 사용)
    df = refresh_analysis_data()

    if not df.empty:
        # 상단에 전체 삭제 버튼 추가
        col_delete, col_refresh, col_info = st.columns([1, 1, 2])

        with col_delete:
            if st.button(
                "🗑️ 전체 데이터 삭제",
                type="secondary",
                help="모든 분석 데이터를 삭제합니다",
            ):
                if st.session_state.get("confirm_delete_all_data", False):
                    success, error = delete_all_analysis_data()
                    if success:
                        if "items_to_delete" in st.session_state:
                            st.session_state.items_to_delete = []
                        if "confirm_delete_all_data" in st.session_state:
                            del st.session_state["confirm_delete_all_data"]
                        if "confirm_delete_selected" in st.session_state:
                            del st.session_state["confirm_delete_selected"]
                        st.success("✅ 모든 데이터가 삭제되었습니다!")
                        st.rerun()
                    else:
                        st.error(f"❌ 삭제 중 오류가 발생했습니다: {error}")
                else:
                    st.session_state["confirm_delete_all_data"] = True
                    st.warning(
                        "⚠️ 정말로 모든 데이터를 삭제하시겠습니까? 다시 한 번 버튼을 클릭하세요."
                    )

        with col_refresh:
            if st.button("🔄 데이터 새로고침", type="secondary"):
                st.rerun()

        with col_info:
            st.write(f"**총 {len(df)}개의 분석 기록**")

        # 기본 통계
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("총 분석 건수", len(df))

        with col2:
            defect_count = len(df[df["judgment"] == "불량"])
            st.metric("불량 건수", defect_count)

        with col3:
            normal_count = len(df[df["judgment"] == "정품"])
            st.metric("정품 건수", normal_count)

        with col4:
            if len(df) > 0:
                defect_rate = (defect_count / len(df)) * 100
                st.metric("불량률", f"{defect_rate:.1f}%")

        # 시계열 차트
        st.subheader("📈 시간별 분석 현황")
        df["date"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["date"])
        daily_stats = (
            df.groupby(df["date"].dt.date)
            .agg({"judgment": "count", "processing_time_seconds": "mean"})
            .rename(columns={"judgment": "count"})
        )

        if not daily_stats.empty:
            fig_timeline = px.line(
                daily_stats.reset_index(),
                x="date",
                y="count",
                title="일별 분석 건수",
                markers=True,
            )
            st.plotly_chart(fig_timeline, use_container_width=True)

        # 판정 결과 분포
        st.subheader("판정 결과 분포")
        col1, col2 = st.columns(2)

        with col1:
            judgment_counts = df["judgment"].value_counts()
            if not judgment_counts.empty:
                fig_pie = px.pie(
                    values=judgment_counts.values,
                    names=judgment_counts.index,
                    title="판정 결과 분포",
                )
                st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            if "defect_type" in df.columns:
                defect_df = df[df["judgment"] == "불량"]
                if not defect_df.empty:
                    defect_counts = defect_df["defect_type"].value_counts()
                    fig_defect = px.bar(
                        x=defect_counts.index,
                        y=defect_counts.values,
                        title="불량 유형 분포",
                    )
                    st.plotly_chart(fig_defect, use_container_width=True)

        # 모델별 성능 비교
        if "model_used" in df.columns:
            st.subheader("모델별 성능 비교")
            model_stats = (
                df.groupby("model_used")
                .agg(
                    {
                        "processing_time_seconds": "mean",
                        "judgment": "count",
                        "confidence_score": "mean",
                    }
                )
                .round(2)
            )

            st.dataframe(model_stats)

        # 상세 로그 테이블
        st.subheader("📋 상세 분석 로그")

        # 필터링 옵션
        col1, col2, col3 = st.columns(3)

        with col1:
            judgment_filter = st.selectbox(
                "판정 결과 필터", options=["전체"] + list(df["judgment"].unique())
            )

        with col2:
            date_filter = st.date_input(
                "날짜 필터 (이후)",
                value=(
                    pd.to_datetime(df["timestamp"]).min().date()
                    if len(df) > 0
                    else datetime.now().date()
                ),
            )

        with col3:
            show_feedback_only = st.checkbox("피드백 있는 항목만")

        # 필터 적용
        filtered_df = df.copy()

        if judgment_filter != "전체":
            filtered_df = filtered_df[filtered_df["judgment"] == judgment_filter]

        if len(filtered_df) > 0:
            filtered_df["timestamp_date"] = pd.to_datetime(
                filtered_df["timestamp"]
            ).dt.date
            filtered_df = filtered_df[filtered_df["timestamp_date"] >= date_filter]

        if show_feedback_only and "user_feedback" in filtered_df.columns:
            filtered_df = filtered_df[
                filtered_df["user_feedback"].notna()
                & (filtered_df["user_feedback"] != "")
            ]

        # 테이블 표시 (개별 삭제 기능 포함)
        if not filtered_df.empty:
            display_columns = [
                "timestamp",
                "filename",
                "judgment",
                "confidence_score",
                "processing_time_seconds",
                "model_used",
                "user_feedback",
            ]
            available_columns = [
                col for col in display_columns if col in filtered_df.columns
            ]

            # 개별 삭제를 위한 체크박스 추가
            st.write("**개별 항목 삭제:**")

            # 삭제할 항목들을 저장할 리스트 초기화
            if "items_to_delete" not in st.session_state:
                st.session_state.items_to_delete = []

            # 전체 선택/해제 및 삭제 버튼
            col_select_all, col_delete_selected, col_clear_selection = st.columns(
                [1, 1, 1]
            )

            with col_select_all:
                select_all = st.checkbox("전체 선택")
                if select_all:
                    # 현재 필터된 데이터의 모든 timestamp 선택
                    st.session_state.items_to_delete = filtered_df["timestamp"].tolist()

            with col_clear_selection:
                if st.button("🔄 선택 해제", type="secondary"):
                    st.session_state.items_to_delete = []
                    if "confirm_delete_selected" in st.session_state:
                        del st.session_state["confirm_delete_selected"]
                    st.rerun()

            with col_delete_selected:
                if st.button("🗑️ 선택된 항목 삭제", type="secondary"):
                    if st.session_state.items_to_delete:
                        if st.session_state.get("confirm_delete_selected", False):
                            # 실제 삭제 실행
                            deleted_count, error = delete_selected_analysis_data(
                                st.session_state.items_to_delete
                            )

                            if error:
                                st.error(f"❌ 삭제 중 오류가 발생했습니다: {error}")
                            else:
                                st.success(
                                    f"✅ {deleted_count}개 항목이 삭제되었습니다!"
                                )

                                # 상태 초기화
                                st.session_state.items_to_delete = []
                                st.session_state["confirm_delete_selected"] = False
                                st.rerun()
                        else:
                            st.session_state["confirm_delete_selected"] = True
                            st.warning(
                                f"⚠️ 정말로 선택된 {len(st.session_state.items_to_delete)}개 항목을 삭제하시겠습니까? 다시 한 번 버튼을 클릭하세요."
                            )
                    else:
                        st.warning("삭제할 항목을 선택해주세요.")

            # 선택된 항목 수 표시
            if st.session_state.items_to_delete:
                st.info(f"📋 선택된 항목: {len(st.session_state.items_to_delete)}개")

            # 각 행에 대한 체크박스와 데이터 표시
            sorted_df = filtered_df[available_columns].sort_values(
                "timestamp", ascending=False
            )

            st.markdown("---")

            for idx, row in sorted_df.iterrows():
                col_check, col_data = st.columns([0.05, 0.95])

                with col_check:
                    # timestamp를 고유 키로 사용
                    item_timestamp = row["timestamp"]

                    is_checked = item_timestamp in st.session_state.items_to_delete

                    if st.checkbox(
                        "",
                        key=f"delete_item_{item_timestamp}",
                        value=is_checked,
                        label_visibility="hidden",
                    ):
                        if item_timestamp not in st.session_state.items_to_delete:
                            st.session_state.items_to_delete.append(item_timestamp)
                    else:
                        if item_timestamp in st.session_state.items_to_delete:
                            st.session_state.items_to_delete.remove(item_timestamp)

                with col_data:
                    # 행 데이터를 사용자 친화적으로 표시
                    col_data1, col_data2 = st.columns([1, 2])

                    with col_data1:
                        st.write(f"**파일명:** {row.get('filename', 'N/A')}")
                        st.write(f"**판정:** {row.get('judgment', 'N/A')}")
                        st.write(f"**모델:** {row.get('model_used', 'N/A')}")

                    with col_data2:
                        st.write(f"**시간:** {row.get('timestamp', 'N/A')}")
                        st.write(
                            f"**처리시간:** {row.get('processing_time_seconds', 'N/A')}초"
                        )
                        if "confidence_score" in row and pd.notna(
                            row["confidence_score"]
                        ):
                            st.write(f"**신뢰도:** {row['confidence_score']}")
                        if (
                            "user_feedback" in row
                            and pd.notna(row["user_feedback"])
                            and row["user_feedback"]
                        ):
                            st.write(f"**피드백:** {row['user_feedback']}")

                st.divider()

            # CSV 다운로드
            st.markdown("---")
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="📥 CSV 다운로드",
                data=csv,
                file_name=f"analysis_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )

        else:
            st.info("필터 조건에 맞는 데이터가 없습니다.")

    else:
        st.info("아직 분석 데이터가 없습니다. 이미지를 업로드하고 분석을 시작해주세요.")

        # 빈 CSV 파일이 있다면 정리
        csv_files = glob.glob(os.path.join(DATA_LOG_FOLDER, "*.csv"))
        for csv_file in csv_files:
            try:
                test_df = pd.read_csv(csv_file, encoding="utf-8")
                if len(test_df) == 0:
                    os.remove(csv_file)
                    st.info(f"빈 데이터 파일 '{csv_file}'를 정리했습니다.")
            except Exception:
                pass


# 추가로 필요한 함수 (기존 코드에 없다면 추가)
def delete_analysis_record(timestamp_to_delete):
    """특정 timestamp의 분석 기록을 삭제하는 함수"""
    try:
        df = load_analysis_logs()
        if df.empty:
            return False, "삭제할 데이터가 없습니다."
        df_filtered = df[df["timestamp"] != timestamp_to_delete]
        df_filtered.to_csv(LOG_CSV_PATH, index=False)
        return True, None
    except Exception as e:
        return False, str(e)


def import_job_to_local(job):
    try:
        jobs = load_finetune_jobs()

        # 이미 로컬에 존재하는지 확인
        if any(local_job["job_id"] == job.id for local_job in jobs):
            st.warning("이미 로컬에 저장된 작업입니다.")
            return

        new_job = {
            "job_id": job.id,
            "model": job.model if hasattr(job, "model") else "",
            "status": job.status if hasattr(job, "status") else "unknown",
            "created_at": job.created_at if hasattr(job, "created_at") else "",
            "fine_tuned_model": getattr(job, "fine_tuned_model", None),
        }

        jobs.append(new_job)
        save_finetune_jobs(jobs)
        st.success(f"✅ 작업 {job.id[:12]}... 가 로컬에 저장되었습니다.")
    except Exception as e:
        st.error(f"작업 로컬 저장 중 오류 발생: {e}")


def fetch_openai_jobs(limit: int = 20):
    """OpenAI에서 파인튜닝 작업 목록을 가져와 세션 상태와 로컬에 저장"""
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise Exception(
                "❌ OPENAI_API_KEY 환경변수가 설정되지 않았습니다. .env 파일을 확인해주세요."
            )

        client = openai.OpenAI(api_key=api_key)
        print("OpenAI에서 파인튜닝 작업 목록을 가져오는 중...")
        response = client.fine_tuning.jobs.list(limit=limit)

        if response.data:
            updated_count = 0
            fetched_jobs = []
            for job in response.data:
                job_data = {
                    "job_id": job.id,
                    "model": job.model,
                    "status": job.status,
                    "created_at": datetime.fromtimestamp(job.created_at).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    "finished_at": (
                        datetime.fromtimestamp(job.finished_at).strftime(
                            "%Y-%m-%d %H:%M:%S"
                        )
                        if job.finished_at
                        else None
                    ),
                    "fine_tuned_model": job.fine_tuned_model,
                    "training_file": job.training_file,
                    "validation_file": job.validation_file,
                    "hyperparameters": (
                        job.hyperparameters.to_dict() if job.hyperparameters else {}
                    ),
                    "result_files": job.result_files,
                    "trained_tokens": job.trained_tokens,
                    "error": job.error.to_dict() if job.error else None,
                    "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
                fetched_jobs.append(job_data)
                if add_finetune_job(job_data):
                    updated_count += 1

            # ✅ 세션 상태에 저장 (전역처럼 사용 가능)
            st.session_state.finetune_jobs_json = fetched_jobs

            # ✅ 로컬 JSON 파일로도 저장
            save_finetune_jobs(fetched_jobs)

            print(
                f"✅ {len(response.data)}개의 작업을 조회했습니다. ({updated_count}개 업데이트)"
            )

            # 상태 통계 출력
            status_counts = {}
            for job in fetched_jobs:
                status = job["status"]
                status_counts[status] = status_counts.get(status, 0) + 1

            if status_counts:
                print("\n📊 작업 상태별 통계:")
                for status, count in status_counts.items():
                    emoji = {
                        "running": "🟡",
                        "succeeded": "🟢",
                        "failed": "🔴",
                        "cancelled": "⚫",
                    }.get(status, "⚪")
                    print(f"  {emoji} {status}: {count}개")

            print(f"\n📋 최근 작업 {min(5, len(fetched_jobs))}개:")
            for i, job in enumerate(fetched_jobs[:5]):
                emoji = {
                    "running": "🟡",
                    "succeeded": "🟢",
                    "failed": "🔴",
                    "cancelled": "⚫",
                }.get(job["status"], "⚪")
                print(
                    f"  {i+1}. {emoji} {job['job_id'][:20]}... | {job['status']} | {job['fine_tuned_model'] or job['model']}"
                )

            return {
                "total_jobs": len(fetched_jobs),
                "updated_jobs": updated_count,
                "status_counts": status_counts,
                "jobs": fetched_jobs,
            }

        else:
            st.session_state.finetune_jobs_json = []
            save_finetune_jobs([])  # 빈 리스트도 저장
            return {"total_jobs": 0, "updated_jobs": 0, "status_counts": {}, "jobs": []}

    except openai.AuthenticationError:
        raise Exception("❌ OpenAI API 키가 유효하지 않습니다.")
    except openai.RateLimitError:
        raise Exception("❌ API 요청 한도를 초과했습니다. 잠시 후 다시 시도해주세요.")
    except openai.APIError as e:
        raise Exception(f"❌ OpenAI API 오류: {str(e)}")
    except Exception as e:
        raise Exception(f"❌ 작업 목록 조회 중 오류 발생: {str(e)}")


def clear_finetune_jobs_cache():
    """
    파인튜닝 작업 목록 캐시를 삭제하는 함수
    """
    try:
        # 캐시 파일 경로들 정의
        cache_files = [
            "finetune_jobs.json",
            "data/finetune_jobs.json",
            "cache/finetune_jobs.json",
            "logs/finetune_jobs.json",
        ]

        deleted_files = []

        # 각 경로에서 캐시 파일 삭제 시도
        for file_path in cache_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    deleted_files.append(file_path)
            except Exception as e:
                st.warning(f"⚠️ {file_path} 삭제 중 오류: {str(e)}")

        # 세션 상태에서도 관련 데이터 제거
        session_keys_to_clear = [
            "finetune_jobs",
            "openai_jobs",
            "cached_jobs",
            "jobs_last_updated",
        ]

        cleared_session_keys = []
        for key in session_keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
                cleared_session_keys.append(key)

        # 결과 메시지
        if deleted_files or cleared_session_keys:
            message_parts = []
            if deleted_files:
                message_parts.append(f"파일 {len(deleted_files)}개 삭제")
            if cleared_session_keys:
                message_parts.append(f"세션 데이터 {len(cleared_session_keys)}개 삭제")

            return True, f"캐시 삭제 완료: {', '.join(message_parts)}"
        else:
            return True, "삭제할 캐시 파일이 없습니다."

    except Exception as e:
        return False, f"캐시 삭제 중 오류 발생: {str(e)}"


def estimate_finetuning_cost(sample_count, model):
    """파인튜닝 비용 추정"""
    # OpenAI 파인튜닝 비용 (2024년 기준, 실제 비용은 확인 필요)
    cost_per_1k = {
        "gpt-3.5-turbo-1106": 0.008,
        "gpt-4-turbo-preview": 0.03,
        "gpt-4o-mini": 0.012,
    }

    base_cost = cost_per_1k.get(model, 0.01)
    estimated_tokens = sample_count * 1000  # 샘플당 평균 토큰 수 추정

    return (estimated_tokens / 1000) * base_cost


def update_all_job_status():
    """모든 작업의 상태를 업데이트합니다."""
    try:
        # 환경변수에서 API 키 가져오기
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise Exception(
                "❌ OPENAI_API_KEY 환경변수가 설정되지 않았습니다. .env 파일을 확인해주세요."
            )

        # 로컬 작업 목록 로드
        jobs = load_finetune_jobs()
        if not jobs:
            print("업데이트할 작업이 없습니다.")
            return {"total_jobs": 0, "updated_jobs": 0, "failed_jobs": 0, "errors": []}

        # OpenAI 클라이언트 생성
        client = openai.OpenAI(api_key=api_key)

        print(f"모든 작업 상태를 업데이트하는 중... (총 {len(jobs)}개 작업)")

        updated_count = 0
        failed_count = 0
        errors = []

        for i, job in enumerate(jobs, 1):
            try:
                job_id = job.get("job_id")
                if not job_id:
                    print(f"  {i}. 작업 ID가 없는 항목 건너뜀")
                    continue

                print(f"  {i}/{len(jobs)}. {job_id[:20]}... 상태 확인 중...")

                # OpenAI에서 최신 상태 조회
                updated_job = client.fine_tuning.jobs.retrieve(job_id)

                # 이전 상태와 비교
                old_status = job.get("status")
                new_status = updated_job.status

                # 작업 데이터 업데이트
                job["status"] = new_status
                job["finished_at"] = (
                    datetime.fromtimestamp(updated_job.finished_at).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                    if updated_job.finished_at
                    else None
                )
                job["fine_tuned_model"] = updated_job.fine_tuned_model
                job["trained_tokens"] = updated_job.trained_tokens
                job["error"] = (
                    updated_job.error.to_dict() if updated_job.error else None
                )
                job["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # 상태 변경 확인
                if old_status != new_status:
                    status_emoji = {
                        "running": "🟡",
                        "succeeded": "🟢",
                        "failed": "🔴",
                        "cancelled": "⚫",
                    }.get(new_status, "⚪")
                    print(
                        f"    ✅ 상태 변경: {old_status} → {status_emoji} {new_status}"
                    )
                else:
                    print(f"    ➖ 상태 유지: {new_status}")

                updated_count += 1

            except openai.APIError as api_error:
                error_msg = f"작업 {job_id[:12]}... API 오류: {str(api_error)}"
                print(f"    ⚠️ {error_msg}")
                errors.append(error_msg)
                failed_count += 1
                continue
            except Exception as e:
                error_msg = f"작업 {job_id[:12]}... 상태 업데이트 실패: {str(e)}"
                print(f"    ⚠️ {error_msg}")
                errors.append(error_msg)
                failed_count += 1
                continue

        # 업데이트된 목록 저장
        try:
            save_finetune_jobs(jobs)
            print(f"✅ {updated_count}개 작업의 상태가 업데이트되었습니다.")

            if failed_count > 0:
                print(f"⚠️ {failed_count}개 작업 업데이트 실패")

            # 상태별 통계 표시
            status_counts = {}
            for job in jobs:
                status = job.get("status", "unknown")
                status_counts[status] = status_counts.get(status, 0) + 1

            if status_counts:
                print(f"\n📊 현재 작업 상태별 통계:")
                for status, count in status_counts.items():
                    status_emoji = {
                        "running": "🟡",
                        "succeeded": "🟢",
                        "failed": "🔴",
                        "cancelled": "⚫",
                    }.get(status, "⚪")
                    print(f"  {status_emoji} {status}: {count}개")

            return {
                "total_jobs": len(jobs),
                "updated_jobs": updated_count,
                "failed_jobs": failed_count,
                "errors": errors,
                "status_counts": status_counts,
            }

        except Exception as save_error:
            raise Exception(f"작업 목록 저장 실패: {str(save_error)}")

    except openai.AuthenticationError:
        raise Exception("❌ OpenAI API 키가 유효하지 않습니다.")
    except openai.RateLimitError:
        raise Exception("❌ API 요청 한도를 초과했습니다. 잠시 후 다시 시도해주세요.")
    except Exception as e:
        raise Exception(f"❌ 상태 업데이트 중 오류 발생: {str(e)}")


def save_job_info(
    job_id,
    base_model,
    sample_count,
    hyperparameters,
    training_file_id,
    validation_file_id=None,
):
    try:
        jobs = load_finetune_jobs()
        new_job = {
            "job_id": job_id,
            "base_model": base_model,
            "sample_count": sample_count,
            "hyperparameters": hyperparameters,
            "training_file_id": training_file_id,
            "validation_file_id": validation_file_id,
            "status": "pending",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        jobs.append(new_job)
        save_finetune_jobs(jobs)
    except Exception as e:
        st.error(f"파인튜닝 작업 정보 저장 중 오류 발생: {e}")


def start_finetuning_process(
    sample_count, base_model, epochs, batch_size, learning_rate
):
    """파인튜닝 프로세스를 시작합니다."""
    try:
        if not st.session_state.get("openai_api_key"):
            st.error("❌ OpenAI API 키가 설정되지 않았습니다.")
            return

        client = openai.OpenAI(api_key=st.session_state.openai_api_key)

        with st.spinner("파인튜닝을 시작하는 중..."):
            # 여기서 실제 파인튜닝 작업을 시작하는 코드를 구현
            # (훈련 파일 업로드, 파인튜닝 작업 생성 등)

            # 예시 작업 데이터 (실제로는 OpenAI API 응답에서 가져옴)
            job_data = {
                "job_id": f"ft-{datetime.now().strftime('%Y%m%d%H%M%S')}",  # 임시 ID
                "model": base_model,
                "status": "running",
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "finished_at": None,
                "fine_tuned_model": None,
                "training_file": "training_data.jsonl",
                "validation_file": None,
                "hyperparameters": {
                    "n_epochs": epochs,
                    "batch_size": batch_size,
                    "learning_rate_multiplier": learning_rate,
                },
                "result_files": [],
                "trained_tokens": None,
                "error": None,
                "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

            # 작업 목록에 추가
            if add_finetune_job(job_data):
                st.success("✅ 파인튜닝이 시작되었습니다!")
                st.info(f"작업 ID: `{job_data['job_id']}`")
                st.rerun()
            else:
                st.error("❌ 작업 정보 저장에 실패했습니다.")

    except Exception as e:
        st.error(f"❌ 파인튜닝 시작 중 오류 발생: {str(e)}")


def log_model_deletion(model_id, success, message):
    """
    모델 삭제 작업을 로그에 기록하는 함수

    Args:
        model_id (str): 모델 ID
        success (bool): 삭제 성공 여부
        message (str): 삭제 결과 메시지
    """
    try:
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "action": "model_deletion",
            "model_id": model_id,
            "success": success,
            "message": message,
            "user_session": st.session_state.get("session_id", "unknown"),
        }

        # 로그 파일 경로
        log_file = "logs/model_deletion.json"

        # 로그 디렉토리 생성
        os.makedirs("logs", exist_ok=True)

        # 기존 로그 로드
        logs = []
        if os.path.exists(log_file):
            try:
                with open(log_file, "r", encoding="utf-8") as f:
                    logs = json.load(f)
            except:
                logs = []  # 파일이 손상된 경우 새로 시작

        # 새 로그 추가
        logs.append(log_entry)

        # 로그 파일 저장 (최근 100개만 유지)
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(logs[-100:], f, ensure_ascii=False, indent=2)

    except Exception as e:
        st.warning(f"로그 기록 중 오류: {str(e)}")


def remove_model_from_cache(model_id):
    """
    로컬 캐시에서 삭제된 모델 정보를 제거하는 함수

    Args:
        model_id (str): 제거할 모델 ID
    """
    try:
        # 파인튜닝 작업 캐시에서 제거
        cache_files = [
            "finetune_jobs.json",
            "data/finetune_jobs.json",
            "cache/finetune_jobs.json",
        ]

        for cache_file in cache_files:
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, "r", encoding="utf-8") as f:
                        jobs = json.load(f)

                    # 해당 모델을 가진 작업들의 상태 업데이트
                    updated = False
                    for job in jobs:
                        if job.get("fine_tuned_model") == model_id:
                            job["model_deleted"] = True
                            job["deleted_at"] = datetime.now().isoformat()
                            updated = True

                    # 변경사항이 있으면 파일 저장
                    if updated:
                        with open(cache_file, "w", encoding="utf-8") as f:
                            json.dump(jobs, f, ensure_ascii=False, indent=2)

                except Exception as e:
                    st.warning(f"캐시 파일 {cache_file} 업데이트 중 오류: {str(e)}")

        # 세션 상태에서도 제거
        if "test_model" in st.session_state and st.session_state.test_model == model_id:
            del st.session_state.test_model

    except Exception as e:
        st.warning(f"캐시에서 모델 정보 제거 중 오류: {str(e)}")


def delete_finetuned_model(model_id):
    """
    OpenAI 파인튜닝된 모델을 삭제하는 함수

    Args:
        model_id (str): 삭제할 모델의 ID

    Returns:
        tuple: (success: bool, message: str)
    """
    try:
        # OpenAI API 키 확인
        api_key = st.session_state.get("openai_api_key")
        if not api_key:
            return False, "OpenAI API 키가 설정되지 않았습니다."

        # OpenAI 클라이언트 초기화
        client = openai.OpenAI(api_key=api_key)

        # 모델 ID 유효성 검사
        if not model_id or not isinstance(model_id, str):
            return False, "유효하지 않은 모델 ID입니다."

        # 모델이 파인튜닝된 모델인지 확인 (보안을 위해)
        if not (model_id.startswith("ft:") or "ft-" in model_id):
            return False, "파인튜닝된 모델만 삭제할 수 있습니다."

        # 모델 존재 여부 확인
        try:
            model_info = client.models.retrieve(model_id)
            if not model_info:
                return False, f"모델 '{model_id}'를 찾을 수 없습니다."
        except Exception as e:
            if "404" in str(e) or "not found" in str(e).lower():
                return False, f"모델 '{model_id}'가 존재하지 않습니다."
            else:
                return False, f"모델 정보 조회 중 오류: {str(e)}"

        # 모델 삭제 실행
        try:
            delete_response = client.models.delete(model_id)

            # 삭제 응답 확인
            if hasattr(delete_response, "deleted") and delete_response.deleted:
                # 삭제 로그 기록
                log_model_deletion(model_id, True, "성공적으로 삭제됨")

                # 로컬 캐시에서도 해당 모델 정보 제거
                remove_model_from_cache(model_id)

                return True, f"모델 '{model_id}'가 성공적으로 삭제되었습니다."
            else:
                log_model_deletion(model_id, False, "삭제 응답이 올바르지 않음")
                return False, f"모델 삭제에 실패했습니다. 응답: {delete_response}"

        except Exception as e:
            error_msg = str(e)

            # 일반적인 오류 메시지 처리
            if "insufficient permissions" in error_msg.lower():
                error_msg = "모델 삭제 권한이 없습니다. API 키 권한을 확인하세요."
            elif "model is currently being used" in error_msg.lower():
                error_msg = (
                    "모델이 현재 사용 중입니다. 사용을 중단한 후 다시 시도하세요."
                )
            elif "rate limit" in error_msg.lower():
                error_msg = "API 요청 한도를 초과했습니다. 잠시 후 다시 시도하세요."

            log_model_deletion(model_id, False, error_msg)
            return False, f"모델 삭제 중 오류 발생: {error_msg}"

    except Exception as e:
        error_msg = f"예상치 못한 오류 발생: {str(e)}"
        log_model_deletion(model_id, False, error_msg)
        return False, error_msg


def bulk_delete_models(model_ids):
    """
    여러 모델을 일괄 삭제하는 함수

    Args:
        model_ids (list): 삭제할 모델 ID 목록

    Returns:
        dict: 삭제 결과 요약
    """
    results = {"success_count": 0, "fail_count": 0, "results": []}

    for model_id in model_ids:
        success, message = delete_finetuned_model(model_id)

        results["results"].append(
            {"model_id": model_id, "success": success, "message": message}
        )

        if success:
            results["success_count"] += 1
        else:
            results["fail_count"] += 1

    return results


def confirm_model_deletion_widget(model_id):
    """
    모델 삭제 확인을 위한 Streamlit 위젯

    Args:
        model_id (str): 삭제할 모델 ID

    Returns:
        bool: 삭제 실행 여부
    """
    st.warning("⚠️ **모델 삭제 확인**")
    st.write(f"모델 ID: `{model_id}`")
    st.write("이 작업은 되돌릴 수 없습니다. 정말로 삭제하시겠습니까?")

    # 확인 입력
    confirm_text = st.text_input(
        "확인을 위해 모델 ID를 다시 입력하세요:", key=f"confirm_delete_{model_id}"
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("❌ 취소", key=f"cancel_delete_{model_id}"):
            return False

    with col2:
        if confirm_text == model_id:
            if st.button(
                "🗑️ 삭제 확인", type="secondary", key=f"execute_delete_{model_id}"
            ):
                return True
        else:
            st.button("🗑️ 삭제 확인", disabled=True, key=f"disabled_delete_{model_id}")

    return False


def get_deletion_logs():
    """
    모델 삭제 로그를 조회하는 함수

    Returns:
        list: 삭제 로그 목록
    """
    try:
        log_file = "logs/model_deletion.json"
        if os.path.exists(log_file):
            with open(log_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return []
    except Exception as e:
        st.error(f"삭제 로그 조회 중 오류: {str(e)}")
        return []


def test_finetuned_model(
    model_name: str, prompt: str, max_tokens: int = 150, temperature: float = 0.7
):
    """파인튜닝된 모델을 테스트합니다."""
    try:
        # 환경변수에서 API 키 가져오기
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise Exception(
                "❌ OPENAI_API_KEY 환경변수가 설정되지 않았습니다. .env 파일을 확인해주세요."
            )

        # OpenAI 클라이언트 생성
        client = openai.OpenAI(api_key=api_key)

        # 모델 응답 생성
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )

        return response.choices[0].message.content

    except openai.APIError as e:
        if "does not exist" in str(e):
            raise Exception(
                f"모델 '{model_name}'을 찾을 수 없습니다. 모델 이름을 확인해주세요."
            )
        elif "insufficient_quota" in str(e):
            raise Exception("API 사용량이 부족합니다. 결제 정보를 확인해주세요.")
        else:
            raise Exception(f"OpenAI API 오류: {str(e)}")
    except Exception as e:
        raise Exception(f"모델 테스트 중 오류가 발생했습니다: {str(e)}")


with tab5:
    st.header("설정")

    # AI 모델 설정
    st.subheader("AI 모델 설정")

    col1, col2 = st.columns(2)

    with col1:
        # 사용할 모델 선택
        available_models = [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-4",
            "gpt-3.5-turbo",
        ]

        # ✅ 파인튜닝된 모델들 추가
        jobs = load_finetune_jobs()  # 이제 리스트를 바로 받음

        finetuned_models = []
        for job in jobs:
            if job.get("status") == "succeeded" and job.get("fine_tuned_model"):
                finetuned_models.append(job["fine_tuned_model"])

        all_models = available_models + finetuned_models

        selected_model = st.selectbox(
            "사용할 모델",
            options=all_models,
            index=(
                all_models.index(st.session_state.get("selected_model", "gpt-4o"))
                if st.session_state.get("selected_model", "gpt-4o") in all_models
                else 0
            ),
        )
        st.session_state.selected_model = selected_model

        if selected_model in finetuned_models:
            st.success("🎯 파인튜닝된 모델을 사용합니다!")

    with col2:
        # 자동 분석 설정
        auto_analysis = st.checkbox(
            "자동 분석 활성화",
            value=st.session_state.get("auto_analysis", False),
            help="새 이미지가 도착하면 자동으로 분석을 수행합니다.",
        )
        st.session_state.auto_analysis = auto_analysis

    # 시스템 프롬프트 설정
    st.subheader("💭 시스템 프롬프트")

    default_prompt = """당신은 이미지를 분석하여 도장 불량률을 판단하는 전문 AI 어시스턴트입니다. 

다음 기준으로 판단해주세요:
- 빨간색 종이로 감싸진 물체: 불량품
- 파란색 종이로 감싸진 물체: 정품
- 기타 색상이나 특이사항이 있는 경우: 상세히 설명

판정 결과와 함께 신뢰도(%)를 제공해주세요."""

    system_prompt = st.text_area(
        "시스템 프롬프트",
        value=st.session_state.get("system_prompt", default_prompt),
        height=200,
        help="AI가 이미지를 분석할 때 사용할 지침을 입력하세요.",
    )
    st.session_state.system_prompt = system_prompt

    # 파일 관리
    st.subheader("📁 파일 관리")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🗑️ 이미지 파일 정리"):
            try:
                # 7일 이상 된 이미지 파일 삭제
                cutoff_date = datetime.now() - timedelta(days=7)
                deleted_count = 0

                for filename in os.listdir(UPLOAD_FOLDER):
                    filepath = os.path.join(UPLOAD_FOLDER, filename)
                    if os.path.isfile(filepath):
                        file_time = datetime.fromtimestamp(os.path.getctime(filepath))
                        if file_time < cutoff_date:
                            os.remove(filepath)
                            deleted_count += 1

                st.success(f"✅ {deleted_count}개의 오래된 이미지 파일을 삭제했습니다.")
            except Exception as e:
                st.error(f"❌ 파일 정리 실패: {e}")

    with col2:
        if st.button("📊 로그 백업"):
            try:
                # 로그 파일 백업
                backup_name = f"analysis_log_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                backup_path = os.path.join(DATA_LOG_FOLDER, backup_name)
                shutil.copy2(LOG_CSV_PATH, backup_path)
                st.success(f"✅ 로그가 백업되었습니다: {backup_name}")
            except Exception as e:
                st.error(f"❌ 로그 백업 실패: {e}")

    with col3:
        # 훈련 데이터 다운로드
        if os.path.exists(TRAINING_DATA_PATH):
            with open(TRAINING_DATA_PATH, "r", encoding="utf-8") as f:
                training_data = f.read()

            st.download_button(
                label="📥 훈련 데이터 다운로드",
                data=training_data,
                file_name="training_data.jsonl",
                mime="application/json",
            )

    # 시스템 정보
    st.subheader("시스템 정보")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**디렉토리 정보:**")
        st.write(f"- 이미지 저장소: {UPLOAD_FOLDER}")
        st.write(f"- 분석 로그: {DATA_LOG_FOLDER}")
        st.write(f"- 파인튜닝 데이터: {FINETUNE_FOLDER}")
        st.write(f"- 파인튜닝 작업: {FINETUNE_JOBS_FOLDER}")

    with col2:
        st.write("**파일 개수:**")
        try:
            image_count = len(
                [
                    f
                    for f in os.listdir(UPLOAD_FOLDER)
                    if f.lower().endswith((".png", ".jpg", ".jpeg"))
                ]
            )
            log_count = len(
                [f for f in os.listdir(DATA_LOG_FOLDER) if f.endswith(".csv")]
            )
            finetune_count = len(
                [f for f in os.listdir(FINETUNE_FOLDER) if f.endswith(".jsonl")]
            )

            st.write(f"- 이미지 파일: {image_count}개")
            st.write(f"- 로그 파일: {log_count}개")
            st.write(f"- 훈련 데이터: {finetune_count}개")

            # 디스크 사용량 정보
            total_size = 0
            for folder in [UPLOAD_FOLDER, DATA_LOG_FOLDER, FINETUNE_FOLDER]:
                for root, dirs, files in os.walk(folder):
                    for file in files:
                        total_size += os.path.getsize(os.path.join(root, file))

            total_size_mb = total_size / (1024 * 1024)
            st.write(f"- 총 사용량: {total_size_mb:.2f} MB")

        except Exception as e:
            st.error(f"시스템 정보 로드 실패: {e}")

    # 고급 설정
    st.subheader("🔧 고급 설정")

    col1, col2 = st.columns(2)

    with col1:
        # 이미지 품질 설정
        image_quality = st.slider(
            "이미지 압축 품질",
            min_value=10,
            max_value=100,
            value=st.session_state.get("image_quality", 85),
            step=5,
            help="업로드된 이미지의 압축 품질을 설정합니다. 높을수록 용량이 커집니다.",
        )
        st.session_state.image_quality = image_quality

        # 최대 이미지 크기 설정
        max_image_size = st.number_input(
            "최대 이미지 크기 (MB)",
            min_value=1,
            max_value=50,
            value=st.session_state.get("max_image_size", 10),
            help="업로드 가능한 최대 이미지 크기를 설정합니다.",
        )
        st.session_state.max_image_size = max_image_size

    with col2:
        # API 요청 타임아웃 설정
        api_timeout = st.number_input(
            "API 타임아웃 (초)",
            min_value=10,
            max_value=300,
            value=st.session_state.get("api_timeout", 60),
            help="OpenAI API 요청의 타임아웃 시간을 설정합니다.",
        )
        st.session_state.api_timeout = api_timeout

        # 최대 분석 기록 수
        max_records = st.number_input(
            "최대 분석 기록 수",
            min_value=100,
            max_value=10000,
            value=st.session_state.get("max_records", 1000),
            step=100,
            help="저장할 최대 분석 기록 수입니다. 초과 시 오래된 기록이 삭제됩니다.",
        )
        st.session_state.max_records = max_records

    # 데이터베이스 관리
    st.subheader("🗄️ 데이터베이스 관리")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🔄 데이터베이스 최적화"):
            try:
                # CSV 파일 최적화 (중복 제거, 정렬)
                if os.path.exists(LOG_CSV_PATH):
                    df = pd.read_csv(LOG_CSV_PATH, encoding="utf-8")
                    # 중복 제거 (타임스탬프 기준)
                    df = df.drop_duplicates(subset=["timestamp"], keep="last")
                    # 최신 순으로 정렬
                    df = df.sort_values("timestamp", ascending=False)
                    # 최대 기록 수 제한
                    df = df.head(st.session_state.get("max_records", 1000))
                    df.to_csv(LOG_CSV_PATH, index=False)
                    st.success("✅ 데이터베이스가 최적화되었습니다.")
                else:
                    st.warning("⚠️ 분석 로그 파일이 없습니다.")
            except Exception as e:
                st.error(f"❌ 데이터베이스 최적화 실패: {e}")

    with col2:
        if st.button("📤 전체 데이터 내보내기"):
            try:
                # 모든 데이터를 ZIP으로 압축
                import zipfile

                zip_name = f"quality_control_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
                zip_path = os.path.join(DATA_LOG_FOLDER, zip_name)

                with zipfile.ZipFile(zip_path, "w") as zipf:
                    # 로그 파일 추가
                    if os.path.exists(LOG_CSV_PATH):
                        zipf.write(LOG_CSV_PATH, "new_analysis_log.csv")

                    # 훈련 데이터 추가
                    if os.path.exists(TRAINING_DATA_PATH):
                        zipf.write(TRAINING_DATA_PATH, "training_data.jsonl")

                    # 최근 이미지 파일들 추가 (최대 100개)
                    image_files = [
                        f
                        for f in os.listdir(UPLOAD_FOLDER)
                        if f.lower().endswith((".png", ".jpg", ".jpeg"))
                    ]
                    image_files.sort(
                        key=lambda x: os.path.getctime(os.path.join(UPLOAD_FOLDER, x)),
                        reverse=True,
                    )

                    for i, img_file in enumerate(image_files[:100]):
                        img_path = os.path.join(UPLOAD_FOLDER, img_file)
                        zipf.write(img_path, f"images/{img_file}")

                # 다운로드 버튼 제공
                with open(zip_path, "rb") as f:
                    st.download_button(
                        label="📥 ZIP 파일 다운로드",
                        data=f.read(),
                        file_name=zip_name,
                        mime="application/zip",
                    )

                # 임시 ZIP 파일 삭제
                os.remove(zip_path)

            except Exception as e:
                st.error(f"❌ 데이터 내보내기 실패: {e}")

    with col3:
        if st.button("⚠️ 모든 데이터 초기화", type="secondary"):
            if st.button("🔴 정말로 초기화하시겠습니까?", type="primary"):
                try:
                    # 모든 파일 삭제
                    for folder in [UPLOAD_FOLDER, DATA_LOG_FOLDER, FINETUNE_FOLDER]:
                        for filename in os.listdir(folder):
                            file_path = os.path.join(folder, filename)
                            if os.path.isfile(file_path):
                                os.remove(file_path)

                    # 세션 상태 초기화
                    for key in list(st.session_state.keys()):
                        if key not in [
                            "selected_model",
                            "system_prompt",
                            "auto_analysis",
                        ]:
                            del st.session_state[key]

                    st.success("✅ 모든 데이터가 초기화되었습니다.")
                    st.rerun()

                except Exception as e:
                    st.error(f"❌ 데이터 초기화 실패: {e}")

    # 설정 저장/로드
    st.subheader("💾 설정 관리")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("💾 현재 설정 저장"):
            try:
                settings = {
                    "selected_model": st.session_state.get("selected_model", "gpt-4o"),
                    "system_prompt": st.session_state.get(
                        "system_prompt", default_prompt
                    ),
                    "auto_analysis": st.session_state.get("auto_analysis", True),
                    "image_quality": st.session_state.get("image_quality", 85),
                    "max_image_size": st.session_state.get("max_image_size", 10),
                    "api_timeout": st.session_state.get("api_timeout", 60),
                    "max_records": st.session_state.get("max_records", 1000),
                }

                settings_path = os.path.join(DATA_LOG_FOLDER, "settings.json")
                with open(settings_path, "w", encoding="utf-8") as f:
                    json.dump(settings, f, ensure_ascii=False, indent=2)

                st.success("✅ 설정이 저장되었습니다.")
            except Exception as e:
                st.error(f"❌ 설정 저장 실패: {e}")

    with col2:
        if st.button("📂 저장된 설정 로드"):
            try:
                settings_path = os.path.join(DATA_LOG_FOLDER, "settings.json")
                if os.path.exists(settings_path):
                    with open(settings_path, "r", encoding="utf-8") as f:
                        settings = json.load(f)

                    # 세션 상태에 설정 적용
                    for key, value in settings.items():
                        st.session_state[key] = value

                    st.success("✅ 저장된 설정이 로드되었습니다.")
                    st.rerun()
                else:
                    st.warning("⚠️ 저장된 설정 파일이 없습니다.")
            except Exception as e:
                st.error(f"❌ 설정 로드 실패: {e}")

    # 앱 정보
    st.subheader("ℹ️ 애플리케이션 정보")

    info_col1, info_col2 = st.columns(2)

    with info_col1:
        st.write("**버전 정보:**")
        st.write("- 앱 버전: v1.0.0")
        st.write("- Streamlit 버전:", st.__version__)
        st.write("- Python 버전:", sys.version.split()[0])

    with info_col2:
        st.write("**기능:**")
        st.write("- 🖼️ 이미지 품질 분석")
        st.write("- 🤖 AI 모델 파인튜닝")
        st.write("- 📊 실시간 대시보드")
        st.write("- 📈 상세 통계 분석")
        st.write("- ⚙️ 고급 설정 관리")


def display_job_details(job: Dict[str, Any]):
    """작업 상세 정보를 표시합니다."""
    try:
        # 기본 정보
        col1, col2 = st.columns(2)

        with col1:
            st.write(f"**작업 ID:** `{job.get('job_id', 'N/A')}`")
            st.write(f"**기본 모델:** {job.get('model', 'N/A')}")
            st.write(f"**생성일시:** {job.get('created_at', 'N/A')}")

        with col2:
            st.write(f"**상태:** {job.get('status', 'N/A')}")
            st.write(f"**완료일시:** {job.get('finished_at', 'N/A')}")
            st.write(f"**파인튜닝 모델:** {job.get('fine_tuned_model', 'N/A')}")

        # 하이퍼파라미터
        hyperparams = job.get("hyperparameters", {})
        if hyperparams:
            st.write("**하이퍼파라미터:**")
            param_cols = st.columns(3)
            with param_cols[0]:
                st.write(f"- 에포크: {hyperparams.get('n_epochs', 'N/A')}")
            with param_cols[1]:
                st.write(f"- 배치 크기: {hyperparams.get('batch_size', 'N/A')}")
            with param_cols[2]:
                st.write(
                    f"- 학습률: {hyperparams.get('learning_rate_multiplier', 'N/A')}"
                )

        # 훈련 정보
        if job.get("trained_tokens"):
            st.write(f"**훈련된 토큰 수:** {job.get('trained_tokens'):,}")

        # 에러 정보
        error = job.get("error")
        if error:
            st.error(f"**오류:** {error.get('message', 'Unknown error')}")

        # 파일 정보
        files_info = []
        if job.get("training_file"):
            files_info.append(f"훈련 파일: {job.get('training_file')}")
        if job.get("validation_file"):
            files_info.append(f"검증 파일: {job.get('validation_file')}")

        if files_info:
            st.write("**파일 정보:**")
            for info in files_info:
                st.write(f"- {info}")

        # 작업 조작 버튼들
        if job.get("status") == "running":
            col1, col2 = st.columns(2)
            with col1:
                if st.button(f"⏹️ 취소", key=f"cancel_{job.get('job_id')}"):
                    cancel_job(job.get("job_id"))
            with col2:
                if st.button(f"🔄 상태 확인", key=f"check_{job.get('job_id')}"):
                    check_job_status(job.get("job_id"))

        elif job.get("status") == "succeeded" and job.get("fine_tuned_model"):
            if st.button(f"🧪 모델 테스트", key=f"test_{job.get('job_id')}"):
                st.session_state.test_model_id = job.get("fine_tuned_model")
                st.info(f"테스트할 모델: {job.get('fine_tuned_model')}")

    except Exception as e:
        st.error(f"작업 정보 표시 중 오류: {str(e)}")


with tab3:
    st.header("파인튜닝 관리")

    # 파인튜닝 설정
    st.subheader("파인튜닝 설정")

    col1, col2 = st.columns(2)

    with col1:
        finetune_epochs = st.number_input(
            "에포크 수",
            min_value=1,
            max_value=10,
            value=st.session_state.get("finetune_epochs", 3),
            help="훈련 반복 횟수 (1-10)",
        )
        st.session_state.finetune_epochs = finetune_epochs

        finetune_batch_size = st.selectbox(
            "배치 크기",
            options=[1, 2, 4, 8, 16],
            index=[1, 2, 4, 8, 16].index(
                st.session_state.get("finetune_batch_size", 1)
            ),
            help="배치 크기가 클수록 안정적이지만 메모리 사용량 증가",
        )
        st.session_state.finetune_batch_size = finetune_batch_size

    with col2:
        finetune_learning_rate = st.number_input(
            "학습률",
            min_value=0.00001,
            max_value=0.1,
            value=st.session_state.get("finetune_learning_rate", 0.0001),
            step=0.00001,
            format="%.5f",
            help="학습률 (일반적으로 0.0001-0.001 사이)",
        )
        st.session_state.finetune_learning_rate = finetune_learning_rate

        base_model = st.selectbox(
            "기본 모델",
            options=["gpt-3.5-turbo-1106", "gpt-4-turbo-preview", "gpt-4o-mini"],
            index=0,
            help="파인튜닝할 기본 모델 선택",
        )

    # 훈련 데이터 준비
    st.subheader("📊 훈련 데이터 준비")

    # 분석 로그 데이터 확인
    try:
        df = load_analysis_logs()
        if not df.empty:
            feedback_count = len(
                df[df["user_feedback"].notna() & (df["user_feedback"] != "")]
            )

            # 데이터 품질 체크
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("총 데이터", len(df))
            with col2:
                st.metric("피드백 데이터", feedback_count)
            with col3:
                if feedback_count >= 10:
                    st.metric("상태", "준비완료", delta="✅")
                else:
                    st.metric("상태", "부족", delta="❌")

            # 최소 요구사항 체크
            if feedback_count < 10:
                st.error(
                    "⚠️ 파인튜닝을 위해서는 최소 10개의 피드백 데이터가 필요합니다."
                )
                st.info(
                    "현재 OpenAI API 요구사항에 따라 최소 10개의 훈련 샘플이 필요합니다."
                )
            elif feedback_count < 50:
                st.warning("💡 더 나은 성능을 위해 50개 이상의 데이터를 권장합니다.")

            if feedback_count >= 10:
                # 데이터 분할 옵션
                train_ratio = st.slider(
                    "훈련/검증 데이터 비율",
                    min_value=0.7,
                    max_value=0.95,
                    value=0.8,
                    step=0.05,
                    help="훈련 데이터 비율 (나머지는 검증 데이터로 사용)",
                )

                if st.button("📦 훈련 데이터 준비", type="primary"):
                    with st.spinner("훈련 데이터를 준비하고 있습니다..."):
                        try:
                            sample_count, message = prepare_training_data()

                            if sample_count > 0:
                                st.success(f"✅ {message}")

                                # 데이터 통계 표시
                                train_count = int(sample_count * train_ratio)
                                val_count = sample_count - train_count

                                col1, col2 = st.columns(2)
                                with col1:
                                    st.info(f"📚 훈련 데이터: {train_count}개")
                                with col2:
                                    st.info(f"🔍 검증 데이터: {val_count}개")

                                # 파인튜닝 시작 섹션
                                st.subheader("🚀 파인튜닝 시작")

                                # 비용 추정
                                estimated_cost = estimate_finetuning_cost(
                                    sample_count, base_model
                                )
                                st.info(f"💰 예상 비용: ${estimated_cost:.2f}")

                                # 확인 체크박스
                                confirm_start = st.checkbox(
                                    "위 설정으로 파인튜닝을 시작하겠습니다.",
                                    help="체크 후 파인튜닝을 시작할 수 있습니다.",
                                )

                                if confirm_start and st.button(
                                    "🚀 파인튜닝 시작", type="secondary"
                                ):
                                    start_finetuning_process(
                                        sample_count,
                                        base_model,
                                        finetune_epochs,
                                        finetune_batch_size,
                                        finetune_learning_rate,
                                    )
                            else:
                                st.error(f"❌ {message}")
                        except Exception as e:
                            st.error(f"❌ 데이터 준비 중 오류 발생: {str(e)}")
                            st.exception(e)  # 디버깅을 위한 상세 오류 표시
        else:
            st.info("분석 로그 데이터가 없습니다. 먼저 텍스트 분석을 수행해주세요.")
    except Exception as e:
        st.error(f"❌ 분석 로그 로드 중 오류: {str(e)}")
        st.exception(e)

    # 파인튜닝 작업 상태
    st.subheader("📋 파인튜닝 작업 현황")

    # 작업 목록 로드 시도
    try:
        jobs = load_finetune_jobs()  # 이미 리스트를 반환한다고 가정

        # 작업이 있는 경우
        if jobs:
            status_filter = st.selectbox(
                "상태 필터",
                options=["전체", "running", "succeeded", "failed", "cancelled"],
                index=0,
            )

            filtered_jobs = (
                jobs
                if status_filter == "전체"
                else [
                    job
                    for job in jobs
                    if isinstance(job, dict) and job.get("status") == status_filter
                ]
            )

            if filtered_jobs:
                st.info(f"📊 총 {len(filtered_jobs)}개의 작업이 있습니다.")

                for job in reversed(filtered_jobs):
                    if not isinstance(job, dict):
                        continue  # 안전하게 무시

                    status_color = {
                        "running": "🟡",
                        "succeeded": "🟢",
                        "failed": "🔴",
                        "cancelled": "⚫",
                    }.get(job.get("status"), "⚪")

                    job_id = job.get("job_id", "Unknown")
                    job_id_display = job_id[:12] + "..." if len(job_id) > 12 else job_id

                    with st.expander(
                        f"{status_color} {job_id_display} ({job.get('status', 'unknown')})",
                        expanded=False,
                    ):
                        display_job_details(job)
            else:
                st.info(f"'{status_filter}' 상태인 작업이 없습니다.")
        else:
            st.info("아직 파인튜닝 작업이 없습니다.")
            st.markdown(
                """
                **처음 사용하시나요?**
                1. 'OpenAI 파인튜닝 작업 목록' 섹션에서 '📋 전체 작업 목록 조회' 버튼을 클릭하여 기존 작업들을 불러올 수 있습니다.
                2. 위의 훈련 데이터 준비 섹션에서 새로운 파인튜닝 작업을 시작할 수 있습니다.
                """
            )

    except Exception as e:
        st.error(f"❌ 파인튜닝 작업 목록 로드 중 오류: {str(e)}")
        st.exception(e)

    # 새로고침 버튼
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("🔄 새로고침", help="작업 목록을 새로고침합니다"):
            st.rerun()

    # OpenAI 파인튜닝 작업 목록 섹션
    st.subheader("🔍 OpenAI 파인튜닝 작업 목록")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📋 전체 작업 목록 조회", type="primary"):
            with st.spinner("OpenAI에서 파인튜닝 작업 목록을 조회하고 있습니다..."):
                try:
                    jobs_data = fetch_openai_jobs()
                    if jobs_data and jobs_data.get("jobs"):
                        job_list = jobs_data["jobs"]
                        st.success(f"✅ {len(job_list)}개의 작업을 찾았습니다.")

                        # 작업 목록만 세션 상태에 저장
                        st.session_state.openai_jobs = job_list

                        # 로컬 파일에는 작업 리스트만 저장
                        save_finetune_jobs(job_list)

                        st.rerun()
                    else:
                        st.info("조회된 파인튜닝 작업이 없습니다.")
                except Exception as e:
                    st.error(f"❌ 작업 목록 조회 중 오류 발생: {str(e)}")
                    st.exception(e)

    with col2:
        if st.button("🗑️ 로컬 캐시 삭제", help="로컬에 저장된 작업 목록을 삭제합니다"):
            try:
                clear_finetune_jobs_cache()
                st.success("✅ 로컬 캐시가 삭제되었습니다.")
                if "openai_jobs" in st.session_state:
                    del st.session_state.openai_jobs
                st.rerun()
            except Exception as e:
                st.error(f"❌ 캐시 삭제 중 오류: {str(e)}")

    # 파인튜닝된 모델 관리
    st.subheader("파인튜닝된 모델 관리")

    # 성공한 작업들에서 모델 목록 추출
    try:
        jobs_data = load_finetune_jobs()

        if isinstance(jobs_data, dict):
            jobs = jobs_data.get("jobs", [])
        elif isinstance(jobs_data, list):
            jobs = jobs_data
        else:
            jobs = []

        successful_jobs = [
            job
            for job in jobs
            if isinstance(job, dict)
            and job.get("status") == "succeeded"
            and job.get("fine_tuned_model")
        ]

        if successful_jobs:
            st.info(f"📊 총 {len(successful_jobs)}개의 파인튜닝된 모델이 있습니다.")

            for job in successful_jobs:
                model_name = job.get("fine_tuned_model", "Unknown")
                model_display = (
                    model_name[:20] + "..." if len(model_name) > 20 else model_name
                )

                with st.expander(f"🤖 {model_display}", expanded=False):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.write(f"**모델 ID:** `{model_name}`")
                        st.write(f"**기본 모델:** {job.get('model', 'Unknown')}")
                        st.write(f"**완료 시간:** {job.get('finished_at', 'Unknown')}")
                        st.write(
                            f"**훈련 파일:** {job.get('training_file', 'Unknown')}"
                        )

                    with col2:
                        # 모델 테스트 버튼
                        if st.button(f"🧪 모델 테스트", key=f"test_{model_name}"):
                            st.session_state.test_model = model_name
                            st.info(f"모델 {model_name}이 테스트용으로 선택되었습니다.")

                        # 모델 삭제 버튼
                        if st.button(
                            f"🗑️ 모델 삭제", key=f"delete_{model_name}", type="secondary"
                        ):
                            st.warning(
                                "⚠️ 모델 삭제는 되돌릴 수 없습니다. 신중하게 진행하세요."
                            )
                            if st.button(
                                f"확인: {model_name} 삭제",
                                key=f"confirm_delete_{model_name}",
                            ):
                                try:
                                    delete_finetuned_model(model_name)
                                    st.success(
                                        f"✅ 모델 {model_name}이 삭제되었습니다."
                                    )
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"❌ 모델 삭제 중 오류: {str(e)}")
        else:
            st.info("아직 완료된 파인튜닝 모델이 없습니다.")

    except Exception as e:
        st.error(f"❌ 모델 목록 로드 중 오류: {str(e)}")

    # 모델 테스트 섹션
    if st.session_state.get("test_model"):
        st.subheader("🧪 모델 테스트")

        test_model = st.session_state.test_model
        st.info(f"테스트 중인 모델: `{test_model}`")

        # 테스트 입력
        test_prompt = st.text_area(
            "테스트 프롬프트",
            placeholder="파인튜닝된 모델을 테스트할 텍스트를 입력하세요...",
            height=100,
        )

        col1, col2 = st.columns(2)

        with col1:
            if st.button("🚀 모델 테스트 실행", type="primary"):
                if test_prompt.strip():
                    with st.spinner("모델 응답을 생성하고 있습니다..."):
                        try:
                            response = test_finetuned_model(test_model, test_prompt)
                            if response:
                                st.success("✅ 모델 응답:")
                                st.write(response)
                        except Exception as e:
                            st.error(f"❌ 모델 테스트 중 오류: {str(e)}")
                else:
                    st.warning("테스트 프롬프트를 입력해주세요.")

        with col2:
            if st.button("❌ 테스트 종료"):
                del st.session_state.test_model
                st.rerun()

    # 도움말 섹션
    with st.expander("❓ 파인튜닝 도움말", expanded=False):
        st.markdown(
            """
        ### 파인튜닝 프로세스
        
        1. **데이터 준비**: 최소 10개의 피드백 데이터가 필요합니다.
        2. **설정 조정**: 에포크 수, 배치 크기, 학습률을 조정합니다.
        3. **훈련 시작**: 설정을 확인하고 파인튜닝을 시작합니다.
        4. **모니터링**: 작업 상태를 주기적으로 확인합니다.
        5. **테스트**: 완료된 모델을 테스트하여 성능을 확인합니다.
        
        ### 권장 설정
        
        - **에포크 수**: 3-5개 (과적합 방지)
        - **배치 크기**: 데이터 크기에 따라 1-8
        - **학습률**: 0.0001-0.001 (보통 0.0001)
        
        ### 주의사항
        
        - 파인튜닝은 시간이 오래 걸릴 수 있습니다 (수십 분~수 시간)
        - 비용이 발생하므로 예상 비용을 확인하세요
        - 품질 좋은 훈련 데이터가 중요합니다
        - 과적합을 방지하기 위해 검증 데이터를 사용하세요
        """
        )

        st.markdown("---")
        st.markdown(
            "**문제가 발생하면 OpenAI API 상태를 확인하고, 로그를 검토하세요.**"
        )


def update_job_status(job_id):
    """OpenAI API에서 특정 파인튜닝 작업 상태를 갱신하여 저장함."""
    try:
        # OpenAI에서 파인튜닝 작업 정보 조회
        response = openai.FineTune.retrieve(id=job_id)

        # 기존 작업 목록 불러오기
        jobs = load_finetune_jobs()

        # 갱신할 작업 정보 생성
        updated_job = {
            "job_id": response.id,
            "status": response.status,
            "created_at": response.created_at,
            "base_model": response.model,
            "fine_tuned_model": getattr(response, "fine_tuned_model", None),
            "training_samples": (
                response.training_files[0].get("bytes", "N/A")
                if response.training_files
                else "N/A"
            ),
            "hyperparameters": (
                response.hyperparameters if hasattr(response, "hyperparameters") else {}
            ),
            # 필요에 따라 더 필드 추가 가능
        }

        # 기존 jobs 목록에서 job_id가 같은 항목을 찾아서 갱신하거나 없으면 추가
        found = False
        for i, job in enumerate(jobs):
            if job.get("job_id") == job_id:
                jobs[i] = updated_job
                found = True
                break
        if not found:
            jobs.append(updated_job)

        # 저장
        save_finetune_jobs(jobs)
        st.success(f"작업 {job_id} 상태가 업데이트 되었습니다.")
    except Exception as e:
        st.error(f"작업 상태 업데이트 실패: {e}")


with tab4:
    st.header("피드백 관리")

    # 강제 새로고침을 위한 세션 상태 초기화
    if "feedback_refresh_trigger" not in st.session_state:
        st.session_state.feedback_refresh_trigger = 0

    if "cached_df" not in st.session_state:
        st.session_state.cached_df = None

    if "last_refresh_time" not in st.session_state:
        st.session_state.last_refresh_time = 0

    # 데이터 로드 함수 (캐싱 포함)
    @st.cache_data(ttl=10)  # 10초간 캐시
    def load_analysis_logs_cached(refresh_trigger):
        try:
            if os.path.exists(LOG_CSV_PATH):
                df = pd.read_csv(LOG_CSV_PATH, encoding="utf-8")
                return df
            else:
                return pd.DataFrame()
        except Exception as e:
            st.error(f"데이터 로드 실패: {str(e)}")
            return pd.DataFrame()

    # 수동 새로고침 버튼
    col_refresh1, col_refresh2, col_refresh3 = st.columns([1, 1, 4])

    with col_refresh1:
        if st.button("🔄 새로고침", help="데이터를 강제로 새로고침합니다"):
            st.session_state.feedback_refresh_trigger += 1
            st.session_state.cached_df = None
            st.cache_data.clear()
            st.rerun()

    with col_refresh2:
        if st.button(
            "🗑️ 전체 삭제", help="모든 분석 기록을 삭제합니다", type="secondary"
        ):
            if st.session_state.get("confirm_delete_all", False):
                try:
                    if os.path.exists(LOG_CSV_PATH):
                        os.remove(LOG_CSV_PATH)
                    # 업로드된 이미지 파일들도 삭제
                    if os.path.exists(UPLOAD_FOLDER):
                        for file in os.listdir(UPLOAD_FOLDER):
                            file_path = os.path.join(UPLOAD_FOLDER, file)
                            if os.path.isfile(file_path):
                                os.remove(file_path)

                    st.session_state.feedback_refresh_trigger += 1
                    st.session_state.cached_df = None
                    st.session_state.confirm_delete_all = False
                    st.cache_data.clear()
                    st.success("✅ 모든 데이터가 삭제되었습니다!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ 삭제 실패: {str(e)}")
            else:
                st.session_state.confirm_delete_all = True
                st.warning("⚠️ 다시 한 번 클릭하면 모든 데이터가 삭제됩니다!")

    # 데이터 로드
    df = load_analysis_logs_cached(st.session_state.feedback_refresh_trigger)

    if not df.empty:
        # 피드백 통계
        st.subheader("📊 피드백 현황")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            total_analyses = len(df)
            st.metric("총 분석 건수", total_analyses)

        with col2:
            feedback_count = len(
                df[df["user_feedback"].notna() & (df["user_feedback"] != "")]
            )
            st.metric("피드백 완료", feedback_count)

        with col3:
            feedback_pending = total_analyses - feedback_count
            st.metric("피드백 대기", feedback_pending)

        with col4:
            if total_analyses > 0:
                feedback_rate = (feedback_count / total_analyses) * 100
                st.metric("피드백 완료율", f"{feedback_rate:.1f}%")

        # 피드백 진행률 시각화
        if total_analyses > 0:
            progress = feedback_count / total_analyses
            st.progress(progress, text=f"피드백 진행률: {progress:.1%}")

        # 필터 옵션
        st.subheader("🔍 분석 결과 필터링")

        col1, col2, col3 = st.columns(3)

        with col1:
            feedback_status = st.selectbox(
                "피드백 상태",
                options=["전체", "피드백 필요", "피드백 완료"],
                index=1,  # 기본값: 피드백 필요
            )

        with col2:
            judgment_filter = st.selectbox(
                "판정 결과",
                options=(
                    ["전체"] + list(df["judgment"].unique())
                    if "judgment" in df.columns
                    else ["전체"]
                ),
            )

        with col3:
            date_range = st.selectbox(
                "기간", options=["전체", "오늘", "최근 3일", "최근 7일", "최근 30일"]
            )

        # 필터 적용
        filtered_df = df.copy()

        # 피드백 상태 필터
        if feedback_status == "피드백 필요":
            filtered_df = filtered_df[
                filtered_df["user_feedback"].isna()
                | (filtered_df["user_feedback"] == "")
            ]
        elif feedback_status == "피드백 완료":
            filtered_df = filtered_df[
                filtered_df["user_feedback"].notna()
                & (filtered_df["user_feedback"] != "")
            ]

        # 판정 결과 필터
        if judgment_filter != "전체" and "judgment" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["judgment"] == judgment_filter]

        # 날짜 필터
        if "timestamp" in filtered_df.columns and date_range != "전체":
            now = datetime.now()
            if date_range == "오늘":
                cutoff = now.replace(hour=0, minute=0, second=0, microsecond=0)
            elif date_range == "최근 3일":
                cutoff = now - timedelta(days=3)
            elif date_range == "최근 7일":
                cutoff = now - timedelta(days=7)
            elif date_range == "최근 30일":
                cutoff = now - timedelta(days=30)

            filtered_df["timestamp"] = pd.to_datetime(filtered_df["timestamp"])
            filtered_df = filtered_df[filtered_df["timestamp"] >= cutoff]

        # 정렬 (최신순)
        if "timestamp" in filtered_df.columns:
            filtered_df = filtered_df.sort_values("timestamp", ascending=False)

        # 피드백 입력 섹션
        st.subheader("피드백 입력")

        if not filtered_df.empty:
            st.write(f"**{len(filtered_df)}개의 결과가 필터 조건에 맞습니다**")

            # 페이지네이션
            items_per_page = 5
            total_pages = (len(filtered_df) - 1) // items_per_page + 1

            if total_pages > 1:
                page = st.selectbox(
                    "페이지 선택",
                    range(1, total_pages + 1),
                    format_func=lambda x: f"{x} 페이지",
                )
                start_idx = (page - 1) * items_per_page
                end_idx = start_idx + items_per_page
                page_df = filtered_df.iloc[start_idx:end_idx]
            else:
                page_df = filtered_df.head(items_per_page)

            # 각 분석 결과에 대한 피드백 입력 폼
            for idx, row in page_df.iterrows():
                with st.expander(
                    f"📸 {row.get('filename', 'Unknown')} - {row.get('judgment', 'Unknown')} "
                    f"({'피드백 완료' if pd.notna(row.get('user_feedback')) and row.get('user_feedback') != '' else '피드백 필요'})",
                    expanded=pd.isna(row.get("user_feedback"))
                    or row.get("user_feedback") == "",
                ):
                    delete_col1, delete_col2 = st.columns([6, 1])

                    with delete_col2:
                        # 안정적인 key 생성
                        delete_key = f"delete_{idx}"
                        confirm_key = f"confirm_delete_{idx}"

                        if st.button(
                            "🗑️ 삭제",
                            key=delete_key,
                            help="이 기록을 삭제합니다",
                            type="secondary",
                        ):
                            if st.session_state.get(confirm_key, False):
                                # 실제 삭제 수행
                                try:
                                    # CSV에서 해당 행 삭제
                                    df_update = pd.read_csv(LOG_CSV_PATH)

                                    # timestamp로 매칭하여 삭제
                                    if "timestamp" in df_update.columns:
                                        mask = (
                                            df_update["timestamp"] == row["timestamp"]
                                        )
                                        df_update = df_update[~mask]

                                        # 이미지 파일 삭제
                                        img_path = None
                                        if "filepath" in row and pd.notna(
                                            row["filepath"]
                                        ):
                                            img_path = row["filepath"]
                                        elif "filename" in row:
                                            possible_path = os.path.join(
                                                UPLOAD_FOLDER, row["filename"]
                                            )
                                            if os.path.exists(possible_path):
                                                img_path = possible_path

                                        if img_path and os.path.exists(img_path):
                                            os.remove(img_path)

                                        # CSV 저장
                                        df_update.to_csv(LOG_CSV_PATH, index=False)

                                        # 세션 상태 초기화 및 새로고침
                                        st.session_state.feedback_refresh_trigger += 1
                                        st.session_state.cached_df = None
                                        if confirm_key in st.session_state:
                                            del st.session_state[confirm_key]
                                        st.cache_data.clear()

                                        st.success("✅ 기록이 삭제되었습니다!")
                                        st.rerun()
                                    else:
                                        st.error(
                                            "❌ 타임스탬프 정보가 없어 삭제할 수 없습니다."
                                        )

                                except Exception as e:
                                    st.error(f"❌ 삭제 실패: {str(e)}")
                            else:
                                st.session_state[confirm_key] = True
                                st.warning("⚠️ 다시 한 번 클릭하면 삭제됩니다!")

                    col1, col2 = st.columns([2, 1])

                    with col1:
                        # 분석 정보 표시
                        st.write("**분석 정보:**")
                        st.write(f"- 파일명: {row.get('filename', 'Unknown')}")
                        st.write(f"- 분석 시간: {row.get('timestamp', 'Unknown')}")
                        st.write(f"- AI 판정: {row.get('judgment', 'Unknown')}")
                        st.write(f"- 신뢰도: {row.get('confidence_score', 'Unknown')}")
                        st.write(
                            f"- 처리 시간: {row.get('processing_time_seconds', 'Unknown')}초"
                        )
                        st.write(f"- 사용 모델: {row.get('model_used', 'Unknown')}")

                        # 상세 분석 결과 표시
                        if "analysis_details" in row and pd.notna(
                            row["analysis_details"]
                        ):
                            st.write("**상세 분석 결과:**")
                            st.write(row["analysis_details"])

                    with col2:
                        # 이미지 표시 (가능한 경우)
                        img_path = None
                        if "filepath" in row and pd.notna(row["filepath"]):
                            img_path = row["filepath"]
                        elif "filename" in row:
                            # UPLOAD_FOLDER에서 이미지 찾기
                            possible_path = os.path.join(UPLOAD_FOLDER, row["filename"])
                            if os.path.exists(possible_path):
                                img_path = possible_path

                        if img_path and os.path.exists(img_path):
                            try:
                                st.image(img_path, caption="분석된 이미지", width=200)
                            except Exception as e:
                                st.write("이미지 로드 실패")

                    # 피드백 입력 폼
                    st.write("---")
                    st.write("**피드백 입력:**")

                    feedback_col1, feedback_col2 = st.columns(2)

                    with feedback_col1:
                        # 정확성 평가
                        accuracy_options = [
                            "선택하세요",
                            "정확함 ✅",
                            "부정확함 ❌",
                            "부분적으로 정확함 ⚠️",
                        ]

                        current_accuracy = "선택하세요"
                        if (
                            pd.notna(row.get("user_feedback"))
                            and row.get("user_feedback") != ""
                        ):
                            for option in accuracy_options[1:]:
                                if option.split()[0] in str(
                                    row.get("user_feedback", "")
                                ):
                                    current_accuracy = option
                                    break

                        accuracy_feedback = st.selectbox(
                            "AI 판정의 정확성",
                            options=accuracy_options,
                            index=accuracy_options.index(current_accuracy),
                            key=f"accuracy_{idx}_{st.session_state.feedback_refresh_trigger}",
                        )

                    with feedback_col2:
                        # 올바른 판정 (AI가 틀린 경우)
                        correct_judgment_options = ["해당없음", "정품", "불량", "기타"]

                        correct_judgment = st.selectbox(
                            "올바른 판정 (AI가 틀린 경우)",
                            options=correct_judgment_options,
                            key=f"correct_{idx}_{st.session_state.feedback_refresh_trigger}",
                        )

                    # 추가 코멘트
                    current_comment = ""
                    if (
                        pd.notna(row.get("user_feedback"))
                        and row.get("user_feedback") != ""
                    ):
                        # 기존 피드백에서 코멘트 부분 추출
                        feedback_parts = str(row.get("user_feedback", "")).split("\n")
                        for part in feedback_parts:
                            if "코멘트:" in part:
                                current_comment = part.replace("코멘트:", "").strip()
                                break

                    additional_comment = st.text_area(
                        "추가 코멘트",
                        value=current_comment,
                        placeholder="분석 결과에 대한 추가적인 의견이나 개선사항을 입력하세요...",
                        key=f"comment_{idx}_{st.session_state.feedback_refresh_trigger}",
                        height=100,
                    )

                    # 피드백 저장 버튼
                    if st.button(
                        f"💾 피드백 저장",
                        key=f"save_{idx}_{st.session_state.feedback_refresh_trigger}",
                    ):
                        if accuracy_feedback != "선택하세요":
                            # 피드백 내용 구성
                            feedback_content = f"정확성: {accuracy_feedback}"

                            if correct_judgment != "해당없음":
                                feedback_content += f"\n올바른 판정: {correct_judgment}"

                            if additional_comment.strip():
                                feedback_content += (
                                    f"\n코멘트: {additional_comment.strip()}"
                                )

                            # CSV 파일 업데이트
                            try:
                                df_update = pd.read_csv(LOG_CSV_PATH)

                                # 해당 행 찾기 (timestamp로 매칭)
                                if "timestamp" in df_update.columns:
                                    mask = df_update["timestamp"] == row["timestamp"]
                                    if mask.any():
                                        df_update.loc[mask, "user_feedback"] = (
                                            feedback_content
                                        )
                                        df_update.loc[mask, "feedback_timestamp"] = (
                                            datetime.now().isoformat()
                                        )

                                        # 올바른 판정이 제공된 경우
                                        if correct_judgment != "해당없음":
                                            df_update.loc[mask, "correct_judgment"] = (
                                                correct_judgment
                                            )

                                        df_update.to_csv(LOG_CSV_PATH, index=False)

                                        # 세션 상태 업데이트
                                        st.session_state.feedback_refresh_trigger += 1
                                        st.session_state.cached_df = None
                                        st.cache_data.clear()

                                        st.success("✅ 피드백이 저장되었습니다!")
                                        st.rerun()
                                    else:
                                        st.error(
                                            "❌ 해당 분석 기록을 찾을 수 없습니다."
                                        )
                                else:
                                    st.error("❌ 타임스탬프 정보가 없습니다.")

                            except Exception as e:
                                st.error(f"❌ 피드백 저장 실패: {str(e)}")
                        else:
                            st.warning("⚠️ 정확성 평가를 선택해주세요.")

            # 대량 피드백 처리
            st.subheader("⚡ 대량 피드백 처리")

            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("✅ 현재 페이지 모두 '정확함'으로 표시"):
                    try:
                        df_update = pd.read_csv(LOG_CSV_PATH)
                        updated_count = 0

                        for idx, row in page_df.iterrows():
                            # 피드백이 없는 경우만 업데이트
                            if (
                                pd.isna(row.get("user_feedback"))
                                or row.get("user_feedback") == ""
                            ):
                                mask = df_update["timestamp"] == row["timestamp"]
                                if mask.any():
                                    df_update.loc[mask, "user_feedback"] = (
                                        "정확성: 정확함 ✅"
                                    )
                                    df_update.loc[mask, "feedback_timestamp"] = (
                                        datetime.now().isoformat()
                                    )
                                    updated_count += 1

                        if updated_count > 0:
                            df_update.to_csv(LOG_CSV_PATH, index=False)

                            # 세션 상태 업데이트
                            st.session_state.feedback_refresh_trigger += 1
                            st.session_state.cached_df = None
                            st.cache_data.clear()

                            st.success(
                                f"✅ {updated_count}개 항목의 피드백이 저장되었습니다!"
                            )
                            st.rerun()
                        else:
                            st.info("ℹ️ 업데이트할 항목이 없습니다.")

                    except Exception as e:
                        st.error(f"❌ 대량 피드백 처리 실패: {str(e)}")

            with col2:
                if st.button("🔄 현재 페이지 피드백 초기화"):
                    try:
                        df_update = pd.read_csv(LOG_CSV_PATH)
                        updated_count = 0

                        for idx, row in page_df.iterrows():
                            mask = df_update["timestamp"] == row["timestamp"]
                            if mask.any():
                                df_update.loc[mask, "user_feedback"] = ""
                                df_update.loc[mask, "feedback_timestamp"] = ""
                                if "correct_judgment" in df_update.columns:
                                    df_update.loc[mask, "correct_judgment"] = ""
                                updated_count += 1

                        if updated_count > 0:
                            df_update.to_csv(LOG_CSV_PATH, index=False)

                            # 세션 상태 업데이트
                            st.session_state.feedback_refresh_trigger += 1
                            st.session_state.cached_df = None
                            st.cache_data.clear()

                            st.success(
                                f"✅ {updated_count}개 항목의 피드백이 초기화되었습니다!"
                            )
                            st.rerun()

                    except Exception as e:
                        st.error(f"❌ 피드백 초기화 실패: {str(e)}")

            with col3:
                if st.button("🗑️ 현재 페이지 모두 삭제"):
                    confirm_batch_delete_key = "confirm_batch_delete"

                    if st.session_state.get(confirm_batch_delete_key, False):
                        try:
                            df_update = pd.read_csv(LOG_CSV_PATH)
                            deleted_count = 0

                            for idx, row in page_df.iterrows():
                                # CSV에서 삭제
                                mask = df_update["timestamp"] == row["timestamp"]
                                df_update = df_update[~mask]

                                # 이미지 파일 삭제
                                img_path = None
                                if "filepath" in row and pd.notna(row["filepath"]):
                                    img_path = row["filepath"]
                                elif "filename" in row:
                                    possible_path = os.path.join(
                                        UPLOAD_FOLDER, row["filename"]
                                    )
                                    if os.path.exists(possible_path):
                                        img_path = possible_path

                                if img_path and os.path.exists(img_path):
                                    os.remove(img_path)

                                deleted_count += 1

                            if deleted_count > 0:
                                df_update.to_csv(LOG_CSV_PATH, index=False)

                                # 세션 상태 초기화
                                st.session_state.feedback_refresh_trigger += 1
                                st.session_state.cached_df = None
                                st.session_state[confirm_batch_delete_key] = False
                                st.cache_data.clear()

                                st.success(
                                    f"✅ {deleted_count}개 항목이 삭제되었습니다!"
                                )
                                st.rerun()

                        except Exception as e:
                            st.error(f"❌ 대량 삭제 실패: {str(e)}")
                    else:
                        st.session_state[confirm_batch_delete_key] = True
                        st.warning(
                            "⚠️ 다시 한 번 클릭하면 현재 페이지의 모든 항목이 삭제됩니다!"
                        )

        else:
            st.info("📝 선택한 필터 조건에 맞는 분석 결과가 없습니다.")

        # 피드백 통계 및 분석
        st.subheader("📈 피드백 분석")

        if feedback_count > 0:
            feedback_df = df[df["user_feedback"].notna() & (df["user_feedback"] != "")]

            col1, col2 = st.columns(2)

            with col1:
                # 정확성 분포
                accuracy_counts = {"정확함": 0, "부정확함": 0, "부분적으로 정확함": 0}

                for feedback in feedback_df["user_feedback"]:
                    if "정확함 ✅" in str(feedback):
                        accuracy_counts["정확함"] += 1
                    elif "부정확함 ❌" in str(feedback):
                        accuracy_counts["부정확함"] += 1
                    elif "부분적으로 정확함 ⚠️" in str(feedback):
                        accuracy_counts["부분적으로 정확함"] += 1

                if sum(accuracy_counts.values()) > 0:
                    fig_accuracy = px.pie(
                        values=list(accuracy_counts.values()),
                        names=list(accuracy_counts.keys()),
                        title="AI 판정 정확성 분포",
                        color_discrete_map={
                            "정확함": "#28a745",
                            "부정확함": "#dc3545",
                            "부분적으로 정확함": "#ffc107",
                        },
                    )
                    st.plotly_chart(fig_accuracy, use_container_width=True)

            with col2:
                # 모델별 정확성
                if "model_used" in feedback_df.columns:
                    model_accuracy = {}

                    for idx, row in feedback_df.iterrows():
                        model = row.get("model_used", "Unknown")
                        feedback = str(row.get("user_feedback", ""))

                        if model not in model_accuracy:
                            model_accuracy[model] = {
                                "정확함": 0,
                                "부정확함": 0,
                                "부분적": 0,
                                "총합": 0,
                            }

                        model_accuracy[model]["총합"] += 1

                        if "정확함 ✅" in feedback:
                            model_accuracy[model]["정확함"] += 1
                        elif "부정확함 ❌" in feedback:
                            model_accuracy[model]["부정확함"] += 1
                        elif "부분적으로 정확함 ⚠️" in feedback:
                            model_accuracy[model]["부분적"] += 1

                    # 정확도 계산 및 표시
                    accuracy_data = []
                    for model, counts in model_accuracy.items():
                        if counts["총합"] > 0:
                            accuracy_rate = (counts["정확함"] / counts["총합"]) * 100
                            accuracy_data.append(
                                {
                                    "모델": model,
                                    "정확도": accuracy_rate,
                                    "총 피드백": counts["총합"],
                                }
                            )

                    if accuracy_data:
                        accuracy_df_display = pd.DataFrame(accuracy_data)
                        fig_model = px.bar(
                            accuracy_df_display,
                            x="모델",
                            y="정확도",
                            title="모델별 정확도",
                            text="총 피드백",
                        )
                        fig_model.update_traces(textposition="outside")
                        fig_model.update_layout(
                            yaxis_title="정확도 (%)", xaxis_title="모델"
                        )
                        st.plotly_chart(fig_model, use_container_width=True)

            # 개선사항 요약
            st.subheader("💡 개선사항 요약")

            # 부정확한 판정들 분석
            incorrect_df = feedback_df[
                feedback_df["user_feedback"].str.contains("부정확함", na=False)
            ]

            if not incorrect_df.empty:
                st.write(f"**부정확한 판정 {len(incorrect_df)}건 발견:**")

                # 올바른 판정 분포 (사용자가 제공한 경우)
                if "correct_judgment" in incorrect_df.columns:
                    correct_judgments = incorrect_df["correct_judgment"].value_counts()
                    if not correct_judgments.empty:
                        st.write("사용자가 제공한 올바른 판정:")
                        for judgment, count in correct_judgments.items():
                            if pd.notna(judgment) and judgment != "":
                                st.write(f"- {judgment}: {count}건")

                # 자주 언급되는 코멘트 키워드
                all_comments = []
                for feedback in incorrect_df["user_feedback"]:
                    if "코멘트:" in str(feedback):
                        comment_part = str(feedback).split("코멘트:")[-1].strip()
                        all_comments.append(comment_part)

                if all_comments:
                    st.write("**주요 개선 피드백:**")
                    # 간단한 키워드 분석 (실제로는 더 정교한 NLP 가능)
                    from collections import Counter
                    import re

                    # 코멘트에서 키워드 추출
                    all_words = []
                    for comment in all_comments:
                        # 한글, 영문, 숫자만 추출하고 2글자 이상인 단어만
                        words = re.findall(r"[가-힣a-zA-Z0-9]{2,}", comment.lower())
                        all_words.extend(words)

                    if all_words:
                        word_counts = Counter(all_words)
                        # 상위 10개 키워드 표시
                        top_keywords = word_counts.most_common(10)

                        for word, count in top_keywords:
                            if count > 1:  # 2번 이상 언급된 키워드만
                                st.write(f"- '{word}': {count}회 언급")

                    # 전체 코멘트 표시 (접기 가능)
                    with st.expander("전체 개선 코멘트 보기"):
                        for i, comment in enumerate(all_comments, 1):
                            st.write(f"{i}. {comment}")

            else:
                st.info("🎉 모든 피드백이 긍정적입니다!")

        # 데이터 내보내기 섹션
        st.subheader("💾 데이터 내보내기")

        col1, col2, col3 = st.columns(3)

        with col1:
            # CSV 다운로드
            if st.button("📥 전체 데이터 CSV 다운로드"):
                try:
                    csv_data = df.to_csv(index=False, encoding="utf-8-sig")
                    st.download_button(
                        label="📄 CSV 파일 다운로드",
                        data=csv_data,
                        file_name=f"analysis_feedback_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                    )
                except Exception as e:
                    st.error(f"❌ CSV 생성 실패: {str(e)}")

        with col2:
            # 피드백만 CSV 다운로드
            if feedback_count > 0:
                if st.button("📥 피드백 데이터만 CSV 다운로드"):
                    try:
                        feedback_only_df = df[
                            df["user_feedback"].notna() & (df["user_feedback"] != "")
                        ]
                        csv_data = feedback_only_df.to_csv(
                            index=False, encoding="utf-8-sig"
                        )
                        st.download_button(
                            label="📄 피드백 CSV 파일 다운로드",
                            data=csv_data,
                            file_name=f"feedback_only_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                        )
                    except Exception as e:
                        st.error(f"❌ 피드백 CSV 생성 실패: {str(e)}")

        with col3:
            # 통계 리포트 다운로드
            if st.button("📊 통계 리포트 다운로드"):
                try:
                    # 통계 리포트 생성
                    report_lines = []
                    report_lines.append("=== 분석 및 피드백 통계 리포트 ===")
                    report_lines.append(
                        f"생성 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    )
                    report_lines.append("")

                    # 기본 통계
                    report_lines.append("📊 기본 통계")
                    report_lines.append(f"- 총 분석 건수: {total_analyses}")
                    report_lines.append(f"- 피드백 완료: {feedback_count}")
                    report_lines.append(f"- 피드백 대기: {feedback_pending}")
                    if total_analyses > 0:
                        report_lines.append(
                            f"- 피드백 완료율: {(feedback_count/total_analyses)*100:.1f}%"
                        )
                    report_lines.append("")

                    # 판정 결과 분포
                    if "judgment" in df.columns:
                        report_lines.append("🎯 판정 결과 분포")
                        judgment_counts = df["judgment"].value_counts()
                        for judgment, count in judgment_counts.items():
                            percentage = (count / total_analyses) * 100
                            report_lines.append(
                                f"- {judgment}: {count}건 ({percentage:.1f}%)"
                            )
                        report_lines.append("")

                    # 정확성 분석 (피드백이 있는 경우)
                    if feedback_count > 0:
                        report_lines.append("✅ 정확성 분석")
                        accuracy_counts = {
                            "정확함": 0,
                            "부정확함": 0,
                            "부분적으로 정확함": 0,
                        }

                        for feedback in feedback_df["user_feedback"]:
                            if "정확함 ✅" in str(feedback):
                                accuracy_counts["정확함"] += 1
                            elif "부정확함 ❌" in str(feedback):
                                accuracy_counts["부정확함"] += 1
                            elif "부분적으로 정확함 ⚠️" in str(feedback):
                                accuracy_counts["부분적으로 정확함"] += 1

                        total_feedback = sum(accuracy_counts.values())
                        if total_feedback > 0:
                            for accuracy, count in accuracy_counts.items():
                                percentage = (count / total_feedback) * 100
                                report_lines.append(
                                    f"- {accuracy}: {count}건 ({percentage:.1f}%)"
                                )
                        report_lines.append("")

                    # 모델별 성능 (해당하는 경우)
                    if "model_used" in df.columns and feedback_count > 0:
                        report_lines.append("🤖 모델별 성능")
                        model_stats = {}

                        for idx, row in feedback_df.iterrows():
                            model = row.get("model_used", "Unknown")
                            feedback = str(row.get("user_feedback", ""))

                            if model not in model_stats:
                                model_stats[model] = {"total": 0, "accurate": 0}

                            model_stats[model]["total"] += 1
                            if "정확함 ✅" in feedback:
                                model_stats[model]["accurate"] += 1

                        for model, stats in model_stats.items():
                            if stats["total"] > 0:
                                accuracy_rate = (
                                    stats["accurate"] / stats["total"]
                                ) * 100
                                report_lines.append(
                                    f"- {model}: {accuracy_rate:.1f}% 정확도 ({stats['accurate']}/{stats['total']})"
                                )
                        report_lines.append("")

                    # 개선사항 요약
                    incorrect_count = len(
                        feedback_df[
                            feedback_df["user_feedback"].str.contains(
                                "부정확함", na=False
                            )
                        ]
                    )
                    if incorrect_count > 0:
                        report_lines.append("💡 개선사항")
                        report_lines.append(f"- 부정확한 판정: {incorrect_count}건")

                        # 올바른 판정 분포
                        if "correct_judgment" in feedback_df.columns:
                            incorrect_df = feedback_df[
                                feedback_df["user_feedback"].str.contains(
                                    "부정확함", na=False
                                )
                            ]
                            correct_judgments = incorrect_df[
                                "correct_judgment"
                            ].value_counts()
                            if not correct_judgments.empty:
                                report_lines.append("- 사용자 제공 올바른 판정:")
                                for judgment, count in correct_judgments.items():
                                    if pd.notna(judgment) and judgment != "":
                                        report_lines.append(
                                            f"  * {judgment}: {count}건"
                                        )
                        report_lines.append("")

                    # 기간별 분석 (최근 7일)
                    if "timestamp" in df.columns:
                        report_lines.append("📅 최근 7일간 활동")
                        now = datetime.now()
                        recent_df = df.copy()
                        recent_df["timestamp"] = pd.to_datetime(recent_df["timestamp"])
                        recent_df = recent_df[
                            recent_df["timestamp"] >= (now - timedelta(days=7))
                        ]

                        if not recent_df.empty:
                            report_lines.append(
                                f"- 최근 7일 분석 건수: {len(recent_df)}"
                            )
                            recent_feedback = len(
                                recent_df[
                                    recent_df["user_feedback"].notna()
                                    & (recent_df["user_feedback"] != "")
                                ]
                            )
                            report_lines.append(
                                f"- 최근 7일 피드백 건수: {recent_feedback}"
                            )
                        else:
                            report_lines.append("- 최근 7일간 활동 없음")

                    # 리포트 문자열 생성
                    report_content = "\n".join(report_lines)

                    st.download_button(
                        label="📄 통계 리포트 다운로드",
                        data=report_content,
                        file_name=f"feedback_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                    )

                except Exception as e:
                    st.error(f"❌ 리포트 생성 실패: {str(e)}")

        # 자동 새로고침 설정
        st.subheader("⚙️ 설정")

        col1, col2 = st.columns(2)

        with col1:
            auto_refresh = st.checkbox(
                "자동 새로고침 활성화 (30초마다)",
                value=st.session_state.get("auto_refresh_enabled", False),
                help="체크하면 30초마다 자동으로 데이터를 새로고침합니다",
            )
            st.session_state.auto_refresh_enabled = auto_refresh

            if auto_refresh:
                # 30초마다 자동 새로고침
                import time

                current_time = time.time()
                if current_time - st.session_state.last_refresh_time > 30:
                    st.session_state.last_refresh_time = current_time
                    st.session_state.feedback_refresh_trigger += 1
                    st.cache_data.clear()
                    st.rerun()

        with col2:
            # 캐시 수동 클리어
            if st.button("🗑️ 캐시 클리어", help="모든 캐시를 수동으로 클리어합니다"):
                st.cache_data.clear()
                st.session_state.cached_df = None
                st.success("✅ 캐시가 클리어되었습니다!")

        # 시스템 정보
        with st.expander("ℹ️ 시스템 정보"):
            st.write(f"**파일 경로:**")
            st.code(f"로그 파일: {LOG_CSV_PATH}")
            st.code(f"업로드 폴더: {UPLOAD_FOLDER}")

            st.write(f"**현재 상태:**")
            st.write(f"- 새로고침 트리거: {st.session_state.feedback_refresh_trigger}")
            st.write(
                f"- 캐시 상태: {'활성' if st.session_state.cached_df is not None else '비활성'}"
            )
            st.write(
                f"- 자동 새로고침: {'활성' if st.session_state.get('auto_refresh_enabled', False) else '비활성'}"
            )

            # 파일 존재 여부 확인
            st.write(f"**파일 상태:**")
            st.write(
                f"- 로그 파일 존재: {'✅' if os.path.exists(LOG_CSV_PATH) else '❌'}"
            )
            st.write(
                f"- 업로드 폴더 존재: {'✅' if os.path.exists(UPLOAD_FOLDER) else '❌'}"
            )

            if os.path.exists(UPLOAD_FOLDER):
                uploaded_files = [
                    f
                    for f in os.listdir(UPLOAD_FOLDER)
                    if os.path.isfile(os.path.join(UPLOAD_FOLDER, f))
                ]
                st.write(f"- 업로드된 파일 수: {len(uploaded_files)}")

    else:
        st.info("📝 아직 분석된 데이터가 없습니다. 먼저 이미지를 분석해보세요!")

        # 빈 상태에서도 시스템 정보는 표시
        with st.expander("ℹ️ 시스템 정보"):
            st.write(f"**파일 경로:**")
            st.code(f"로그 파일: {LOG_CSV_PATH}")
            st.code(f"업로드 폴더: {UPLOAD_FOLDER}")

            st.write(f"**파일 상태:**")
            st.write(
                f"- 로그 파일 존재: {'✅' if os.path.exists(LOG_CSV_PATH) else '❌'}"
            )
            st.write(
                f"- 업로드 폴더 존재: {'✅' if os.path.exists(UPLOAD_FOLDER) else '❌'}"
            )

# 페이지 하단에 사용법 안내
st.markdown("---")
st.markdown(
    """
### 사용법 안내

**피드백 관리 기능:**
1. **필터링**: 피드백 상태, 판정 결과, 기간별로 데이터를 필터링할 수 있습니다
2. **개별 피드백**: 각 분석 결과에 대해 정확성 평가와 코멘트를 입력할 수 있습니다
3. **대량 처리**: 현재 페이지의 모든 항목에 대해 일괄 처리가 가능합니다
4. **데이터 관리**: 개별 삭제, 대량 삭제, 피드백 초기화 기능을 제공합니다 
+ 피드백 관리 창에서 데이터 삭제는 전부 삭제고, 데이터 분석 창 데이터 삭제는 그 탭의 데이터만 삭제됩니다.

**통계 및 분석:**
- 피드백 현황과 정확도를 시각적으로 확인할 수 있습니다
- 모델별 성능 비교가 가능합니다
- 개선사항을 자동으로 요약해줍니다

**데이터 내보내기:**
- 전체 데이터 또는 피드백 데이터만 CSV로 다운로드할 수 있습니다
- 상세한 통계 리포트를 텍스트 파일로 생성할 수 있습니다

💡 **팁**: 자동 새로고침을 활성화하면 실시간으로 데이터 변경사항을 확인할 수 있습니다.
"""
)
```
