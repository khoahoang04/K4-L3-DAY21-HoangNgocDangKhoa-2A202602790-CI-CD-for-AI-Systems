# Báo Cáo Lab Day 21 - CI/CD cho AI Systems

| | |
|---|---|
| Họ và tên | Hoàng Ngọc Đăng Khoa |
| MSSV | 2A202602790 |
| Lớp / Khóa | K4 |
| Repo GitHub | https://github.com/khoahoang04/K4-L3-DAY21-HoangNgocDangKhoa-2A202602790-CI-CD-for-AI-Systems |
| Ngày nộp | 07/10/2026 |

---

## 1. Bộ Siêu Tham Số Đã Chọn và Lý Do

| Lần chạy | n_estimators | learning_rate | max_depth | f1_score | accuracy |
|---|---|---|---|---|---|
| 1 | 200 | 0.1 | 5 | 0.7149 | 0.8740 |
| 2 | 100 | 0.1 | 3 | 0.7109 | 0.8780 |
| 3 | 50 | 0.05 | 2 | 0.6051 | 0.8460 |

**Bộ siêu tham số đã chọn:** `n_estimators=200`, `learning_rate=0.1`, `max_depth=5`.

**Lý do:** Bộ tham số này đạt f1_score cao nhất (0.7149) trên tập holdout. Dù lần chạy 2 có accuracy cao hơn (0.8780 so với 0.8740), f1_score của nó lại thấp hơn (0.7109), phản ánh xu hướng đoán lệch về lớp đa số. Khi tăng n_estimators lên 200 kết hợp max_depth=5, mô hình khai thác tốt hơn các tương tác phi tuyến mà không bị quá khớp. Giữ learning_rate ở 0.1 giúp tốc độ hội tụ ổn định và tối ưu điểm F1 của lớp thiểu số.

---

## 2. Vì Sao Ngưỡng Chất Lượng Đặt Trên F1 Chứ Không Phải Accuracy

Tập dữ liệu Adult mất cân bằng nghiêm trọng khi lớp thu nhập cao (>50K) chỉ chiếm khoảng 24%, còn lớp thu nhập thấp chiếm 76%. Một mô hình ngây thơ luôn dự đoán thu nhập thấp sẽ đạt ngay accuracy 76% nhưng hoàn toàn vô dụng vì bỏ sót toàn bộ khách hàng mục tiêu. Ngược lại, F1 của lớp dương là trung bình điều hòa giữa Precision và Recall của nhóm thu nhập cao, phản ánh chính xác khả năng nhận diện lớp thiểu số mà không bị lớp đa số lấn át. Ta không dùng trung bình macro hay weighted vì việc tính gộp lớp đa số sẽ che giấu hiệu năng thực tế trên lớp cần dự đoán.

---

## 3. Khó Khăn Gặp Phải và Cách Giải Quyết

| Khó khăn | Nguyên nhân | Cách giải quyết |
|---|---|---|
| Job Release lỗi 503 khi kiểm tra health check | VM cài `scikit-learn 1.7.2` không tương thích unpickle với mô hình train từ bản `1.4.2` | Cài đặt chính xác `scikit-learn==1.4.2` trên VM để đồng bộ với môi trường huấn luyện |
| DVC trên GitHub Actions runner nguy cơ thiếu khóa | File cấu hình DVC trỏ khóa cục bộ trong khi GitHub Actions lưu credential tại `/tmp` | Thiết lập workflow ghi secret ra cả đường dẫn cục bộ lẫn biến môi trường Google |
| Lệnh `git push` bị remote từ chối Internal Server Error | Máy chủ GitHub quá tải tạm thời khi xử lý delta | Thực hiện retry lệnh đẩy mã nguồn sau vài giây và push thành công |

---

## 4. So Sánh Bước 2 và Bước 3 (bắt buộc, 2 - 3 câu)

| | f1_score | accuracy |
|---|---|---|
| Bước 2 (chỉ `train_batch1`) | 0.7149 | 0.8740 |
| Bước 3 (thêm `train_batch2`) | 0.7354 | 0.8820 |

**Nhận xét:** Khi bổ sung thêm 22.361 mẫu nâng tổng quy mô lên 44.722 mẫu, F1-score tăng nhẹ từ 0.7149 lên 0.7354 và accuracy tăng từ 0.8740 lên 0.8820. Điểm F1 không tăng đột biến do hai tập dữ liệu lấy từ cùng nguồn khảo sát dân số nên có chung phân phối, mô hình đã học phần lớn đặc trưng cốt lõi từ batch đầu. Quy trình CI/CD tự động đã vận hành trơn tru từ phát hiện thay đổi DVC đến khi cập nhật mô hình lên VM.

---

## 5. Phần Bonus Đã Thực Hiện (nếu có)

- [x] Bonus 2 - Điều chỉnh ngưỡng quyết định: Quét ngưỡng xác suất [0.1, 0.9] tìm ra ngưỡng tối ưu 0.30 nâng F1-score từ 0.7354 lên 0.7537 (+0.018) so với ngưỡng 0.5 mặc định.
- [x] Bonus 3 - Báo cáo precision / recall tự động: Xuất Confusion Matrix và Precision/Recall từng lớp ra `outputs/detail.txt`; bỏ sót người thu nhập cao (Recall thấp) tốn kém hơn vì bỏ lỡ khách hàng tiềm năng.
- [x] Bonus 5 - Cảnh báo lệch lạc dữ liệu: Kiểm tra tỷ lệ lớp dương trong tập train (đạt 24.78%, sát mốc chuẩn 24.8%), cảnh báo khi lệch quá 5% và lưu `positive_ratio` vào `report.json`.
