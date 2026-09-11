import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import shap

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="Churn Prediction App", page_icon="📊", layout="centered")
st.title("🔮 Dự đoán Khách hàng Rời bỏ (Telco Churn)")
st.markdown("Nhập thông tin khách hàng để xem tỷ lệ rủi ro hủy dịch vụ.")

# --- 1. TẢI MÔ HÌNH VÀ CẤU TRÚC CỘT ---
@st.cache_resource # Giúp load mô hình 1 lần duy nhất để app chạy nhanh
def load_model():
    # Lấy thư mục hiện tại của file app.py (tức là thư mục src/)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Nối đường dẫn tuyệt đối đến các file .pkl
    model_path = os.path.join(current_dir, 'xgboost_churn_model.pkl')
    cols_path = os.path.join(current_dir, 'model_columns.pkl')
    
    # Load file
    model = joblib.load(model_path)
    cols = joblib.load(cols_path)
    return model, cols

xgb_model, model_columns = load_model()

# --- 2. GIAO DIỆN NHẬP LIỆU (USER INPUTS) ---
st.header("📋 Thông tin Khách hàng")
col1, col2 = st.columns(2)

with col1:
    tenure = st.number_input("Thâm niên sử dụng (Tháng)", min_value=0, max_value=72, value=12)
    monthly_charges = st.number_input("Cước phí hàng tháng ($)", min_value=15.0, max_value=120.0, value=70.0)
    total_charges = st.number_input("Tổng cước phí ($)", min_value=15.0, max_value=8600.0, value=840.0)

with col2:
    contract = st.selectbox("Loại hợp đồng", ["Month-to-month", "One year", "Two year"])
    internet = st.selectbox("Dịch vụ Internet", ["Fiber optic", "DSL", "No"])

# --- 3. XỬ LÝ DỮ LIỆU ĐỂ DỰ ĐOÁN (DATA PROCESSING) ---
if st.button("🚀 Phân Tích Rủi Ro", type="primary"):
    
    # Tạo một dictionary chứa tất cả các cột của mô hình, mặc định bằng 0
    input_data = {col: 0 for col in model_columns}
    
    # Gán các giá trị số nguyên/thực
    input_data['tenure'] = tenure
    input_data['MonthlyCharges'] = monthly_charges
    input_data['TotalCharges'] = total_charges
    
    # Xử lý One-Hot Encoding cho các biến phân loại
    # Cấu trúc tên cột One-Hot thường là: TênCột_GiáTrị
    contract_col = f"Contract_{contract}"
    if contract_col in input_data:
        input_data[contract_col] = 1
        
    internet_col = f"InternetService_{internet}"
    if internet_col in input_data:
        input_data[internet_col] = 1

    # Chuyển đổi thành DataFrame với 1 dòng duy nhất
    input_df = pd.DataFrame([input_data])
    
    # --- 4. DỰ ĐOÁN VÀ HIỂN THỊ ---
    # predict_proba trả về mảng 2 chiều [xác_suất_0, xác_suất_1]
    churn_prob = xgb_model.predict_proba(input_df)[0][1] * 100
    
    st.subheader("💡 Kết quả phân tích:")
    
    if churn_prob > 50:
        st.error(f"⚠️ Rủi ro Rời bỏ CAO: {churn_prob:.1f}%")
        st.write("**Đề xuất:** Cần gửi ngay voucher giảm giá hoặc gọi điện chăm sóc, mời chuyển sang hợp đồng 1 năm.")
    else:
        st.success(f"✅ Rủi ro Rời bỏ THẤP: {churn_prob:.1f}%")
        st.write("Khách hàng đang hài lòng với dịch vụ. Hãy tiếp tục duy trì!")
        
    # --- 5. GIẢI THÍCH MÔ HÌNH VỚI SHAP ---
    st.divider() # Tạo đường kẻ ngang phân cách
    st.subheader("🧠 Giải thích chi tiết từ Trí tuệ Nhân tạo (SHAP)")
    st.markdown("Biểu đồ dưới đây thể hiện lý do tại sao mô hình đưa ra quyết định này. Màu đỏ làm **tăng** rủi ro rời bỏ, màu xanh làm **giảm** rủi ro.")
        
    # Khởi tạo bộ giải thích SHAP cho mô hình XGBoost
    explainer = shap.Explainer(xgb_model)
        
    # Tính toán giá trị SHAP cho dòng dữ liệu khách hàng hiện tại
    shap_values = explainer(input_df)
        
    # Tạo khung vẽ biểu đồ (Figure) để nhúng vào Streamlit
    fig, ax = plt.subplots(figsize=(8, 4))
        
    # Vẽ biểu đồ thác nước (Waterfall plot)
    # Lấy shap_values[0] vì input_df chỉ có 1 dòng (1 khách hàng)
    shap.plots.waterfall(shap_values[0], show=False)
        
    # Tùy chỉnh layout để chữ không bị cắt
    plt.tight_layout()
        
    # Hiển thị biểu đồ lên Streamlit
    st.pyplot(fig)
