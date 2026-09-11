import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import shap
import plotly.graph_objects as go

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

# --- KHỐI 3: TẠO TAB GIAO DIỆN ---
tab1, tab2 = st.tabs(["👤 Dự đoán Cá nhân", "📁 Dự đoán Hàng loạt (CSV)"])

# ==========================================
# TAB 1: DỰ ĐOÁN CÁ NHÂN (CODE CŨ CỦA BẠN)
# ==========================================
with tab1:
    st.header("📋 Thông tin Khách hàng")
    col1, col2 = st.columns(2)

    with col1:
        tenure = st.number_input("Thâm niên sử dụng (Tháng)", min_value=0, max_value=72, value=12)
        monthly_charges = st.number_input("Cước phí hàng tháng ($)", min_value=15.0, max_value=120.0, value=70.0)
        total_charges = st.number_input("Tổng cước phí ($)", min_value=15.0, max_value=8600.0, value=840.0)

    with col2:
        contract = st.selectbox("Loại hợp đồng", ["Month-to-month", "One year", "Two year"])
        internet = st.selectbox("Dịch vụ Internet", ["Fiber optic", "DSL", "No"])

    if st.button("🚀 Phân Tích Rủi Ro", type="primary", key="single_predict"):
        # --- 4. DỰ ĐOÁN VÀ HIỂN THỊ GAUGE CHART ---
        churn_prob = xgb_model.predict_proba(input_df)[0][1] * 100
            
        st.subheader("💡 Kết quả phân tích Rủi ro")
            
        # Khởi tạo biểu đồ Gauge
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = churn_prob,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Xác suất Hủy dịch vụ", 'font': {'size': 18}},
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "#EF553B" if churn_prob > 50 else "#00CC96"}, # Đỏ nếu rủi ro cao, xanh nếu thấp
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 33], 'color': "rgba(0, 204, 150, 0.15)"},    # Vùng an toàn
                    {'range': [33, 66], 'color': "rgba(255, 236, 0, 0.15)"},   # Vùng cảnh báo
                    {'range': [66, 100], 'color': "rgba(239, 85, 59, 0.15)"}], # Vùng nguy hiểm
                'threshold': {
                    'line': {'color': "black", 'width': 3},
                    'thickness': 0.75,
                    'value': churn_prob}
            }
        ))
            
        # Tùy chỉnh kích thước để biểu đồ gọn gàng hơn
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
            
        # Hiển thị biểu đồ lên Streamlit
        st.plotly_chart(fig, use_container_width=True)
            
        # Nội suy đề xuất động dựa trên input
        if churn_prob > 50:
            if internet == 'Fiber optic':
                st.warning("Đề xuất: Cần cử kỹ thuật viên kiểm tra đường truyền cáp quang của khách hàng này ngay lập tức.")
            elif tenure < 12:
                st.warning("Đề xuất: Khách hàng mới có rủi ro cao. Hãy gửi voucher giảm giá 20% cho tháng cước tiếp theo.")
            else:
                st.warning("Đề xuất: Gọi điện CSKH để tìm hiểu lý do và mời chuyển sang hợp đồng dài hạn.")
        else:
            st.success("Tình trạng ổn định: Hãy duy trì chất lượng dịch vụ hiện tại.")
            
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
        pass

# ==========================================
# TAB 2: DỰ ĐOÁN HÀNG LOẠT (TÍNH NĂNG MỚI)
# ==========================================
with tab2:
    st.header("📂 Tải lên danh sách khách hàng")
    st.markdown("Vui lòng tải lên file `.csv` chứa các cột: `CustomerID` (tùy chọn), `tenure`, `MonthlyCharges`, `TotalCharges`, `Contract`, `InternetService`.")
    
    # Nút upload file
    uploaded_file = st.file_uploader("Chọn file CSV", type=['csv'])
    
    if uploaded_file is not None:
        import numpy as np
        
        # Đọc dữ liệu khách hàng tải lên
        batch_df = pd.read_csv(uploaded_file)
        st.write("👀 Preview dữ liệu tải lên:")
        st.dataframe(batch_df.head(3))
        
        if st.button("🚀 Phân Tích Hàng Loạt", type="primary", key="batch_predict"):
            with st.spinner('Đang xử lý dữ liệu bằng Trí tuệ Nhân tạo...'):
                result_df = batch_df.copy()
                
                # Tạo một DataFrame chứa toàn số 0 với cấu trúc cột chuẩn của mô hình
                input_matrix = pd.DataFrame(0, index=np.arange(len(batch_df)), columns=model_columns)
                
                # Điền dữ liệu dạng số
                for col in ['tenure', 'MonthlyCharges', 'TotalCharges']:
                    if col in batch_df.columns:
                        # Ép kiểu dữ liệu tránh lỗi khoảng trắng
                        input_matrix[col] = pd.to_numeric(batch_df[col], errors='coerce').fillna(0)
                        
                # Xử lý One-Hot Encoding tự động cho hàng loạt dòng
                if 'Contract' in batch_df.columns:
                    for idx, val in batch_df['Contract'].items():
                        col_name = f"Contract_{val}"
                        if col_name in model_columns:
                            input_matrix.at[idx, col_name] = 1
                            
                if 'InternetService' in batch_df.columns:
                    for idx, val in batch_df['InternetService'].items():
                        col_name = f"InternetService_{val}"
                        if col_name in model_columns:
                            input_matrix.at[idx, col_name] = 1
                            
                # Dự đoán xác suất cho toàn bộ danh sách
                probs = xgb_model.predict_proba(input_matrix)[:, 1] * 100
                
                # Gắn kết quả dự đoán vào bảng hiển thị
                result_df['Xác suất Rời bỏ (%)'] = np.round(probs, 2)
                result_df['Cảnh báo'] = result_df['Xác suất Rời bỏ (%)'].apply(lambda x: '🔴 Nguy hiểm' if x > 50 else '🟢 An toàn')
                
                # Lọc ra nhóm khách hàng rủi ro cao để hiển thị ưu tiên
                high_risk_df = result_df[result_df['Xác suất Rời bỏ (%)'] > 50].sort_values(by='Xác suất Rời bỏ (%)', ascending=False)
                
                st.subheader(f"⚠️ Đã phát hiện {len(high_risk_df)} khách hàng có rủi ro cao!")
                st.dataframe(high_risk_df)
                
                # Tạo nút tải file báo cáo định dạng CSV (hỗ trợ tiếng Việt với utf-8-sig)
                csv_data = result_df.to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    label="⬇️ Tải xuống Báo cáo Tổng hợp (CSV)",
                    data=csv_data,
                    file_name="Bao_cao_Du_doan_Churn.csv",
                    mime="text/csv",
                )
