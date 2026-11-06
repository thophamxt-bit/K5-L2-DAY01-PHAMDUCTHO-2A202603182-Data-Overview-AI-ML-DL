# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 2026-11-06

**Runtime Colab:** CPU

**Python / PyTorch / Ultralytics:** Python: 3.13.18 / PyTorch: 2.11.0+cpu / Ultralytics: 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. **Chép record hạng 1:** `class_id`=468 ; `class_name`=cab ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`=ImageNet-1K
2. **Record này nói về cả bức ảnh hay về một vật thể riêng trong ảnh? Vì sao bạn nghĩ vậy?**  
   Record này nói về **cả bức ảnh** (image-level classification). Lý do là bài toán phân loại ảnh không sinh ra tọa độ định vị vị trí không gian (bounding box hay segmentation mask) cho từng vật thể riêng lẻ, mà chỉ gán một phân phối xác suất nhãn cho toàn bộ khung hình ảnh.
3. **Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?**  
   Danh sách tên lớp đi kèm với **taxonomy/checkpoint** định sẵn của tập dữ liệu huấn luyện (cụ thể là tập nhãn chuẩn ImageNet-1K gồm 1000 lớp cố định). Model không tự suy nghĩ hay tự sáng tạo ra tên lớp mới ngoài không gian nhãn đã được thiết lập.
4. **Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?**  
   Cần ghi cả 3 thông tin để:
   - Định danh duy nhất, tránh nhập nhằng ngữ nghĩa (cùng tên lớp có thể mang ý nghĩa hoặc phạm vi khác nhau giữa các bộ dữ liệu).
   - Đảm bảo tính ánh xạ chuẩn xác giữa mã số máy tính đọc (`class_id`) và nhãn người đọc (`class_name`).
   - Xác định rõ không gian phân loại (`taxonomy_name`), tránh nhầm lẫn khi chuyển đổi giữa các checkpoint dùng các taxonomy khác nhau (như ImageNet-1K, COCO-80, OpenImages).
5. **Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn một lớp?**  
   Guideline cần quy định rõ **tiêu chí ưu tiên chủ thể chính (salient object)**:
   - Dựa trên diện tích chiếm phần lớn khung hình (dominant subject).
   - Dựa vào vật thể nằm ở vị trí trung tâm, rõ nét (in-focus).
   - Thiết lập bảng thứ tự ưu tiên nghiệp vụ rõ ràng khi có nhiều đối tượng đồng xuất hiện.
6. **Vì sao không được lấy `score` của model làm ground truth?**  
   Vì `score` chỉ là xác suất thống kê/độ tự tin nội bộ của mô hình, mô hình hoàn toàn có thể tự tin nhưng đoán sai lệch thực tế (overconfident error) hoặc dự đoán sai ngữ cảnh (ví dụ ảnh giao thông đông đúc xe buýt nhưng model lại gán `cab` với score ~0.51). Ground truth bắt buộc phải là sự thật khách quan do con người kiểm chứng và xác nhận theo guideline chuẩn.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. **Chép một record:** `class_name`=person ; `score`=0.91 ; `bbox_xyxy`=[382.45, 172.50, 497.10, 423.80] ; `bbox_width`=114.65 ; `bbox_height`=251.30
2. **Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?**  
   Bounding box có góc trên-trái nằm tại tọa độ pixel $(x=382.45, y=172.50)$, góc dưới-phải nằm tại tọa độ pixel $(x=497.10, y=423.80)$. Hộp có chiều rộng là $114.65$ pixel và chiều cao là $251.30$ pixel, bao trọn vẹn người đầu bếp đứng quay lưng ở trung tâm gian bếp.
3. **Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?**  
   - Ngưỡng **0.20**: có **17** predictions
   - Ngưỡng **0.60**: có **6** predictions
4. **Khi hạ ngưỡng (giữ nhiều hộp hơn), người reviewer phải làm thêm việc gì?**  
   Reviewer phải mất thêm thời gian kiểm tra, phát hiện và xóa bỏ các **dương tính giả (False Positives)**, lọc các bounding box nhiễu/sai nhãn hoặc xử lý các box bị trùng lặp do thuật toán NMS chưa lọc hết.
5. **Viết một quy tắc box để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).**  
   *Bounding box phải bao khít toàn bộ các điểm nhìn thấy được của đối tượng (tight bounding box); không chừa khoảng trống nền thừa quá 2 pixel và không cắt phạm vào biên của vật thể.*
6. **Trên ảnh `kitchen`, chỉ ra một vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?**  
   - *Vật thể quan sát được:* Cánh tay/phần thân người ở sát mép trái khung hình (mô hình phát hiện là `person 0.61`).
   - *Lựa chọn & Giải thích:* Tôi chọn **viết thành quy tắc trong guideline** (ví dụ quy định: *"Vật thể bị cắt mép ảnh chỉ gán nhãn nếu phần nhìn thấy vượt quá 20% diện tích và nhận diện rõ lớp"*). Việc chuẩn hóa trường hợp cắt mép thông thường vào guideline giúp đội ngũ annotator thực hiện đồng bộ mà không cần gửi escalation gây tắc nghẽn quy trình.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. **Chép một record:** `instance_id`=traffic-001 ; `class_name`=bus ; `score`=0.925745 ; số điểm polygon=`polygon_point_count`=120 ; 3–5 điểm đầu của `polygon_xy`=[[148.0, 189.0], [147.0, 190.0], [145.0, 190.0], [143.0, 192.0], [142.0, 192.0]]
2. **So với box, polygon cho thêm thông tin gì về hình dạng vật thể?**  
   Polygon cung cấp đường biên hình học chính xác ở cấp độ pixel (pixel-level mask), mô tả trọn vẹn đường cong, góc cạnh thực tế của đối tượng và loại bỏ hoàn toàn phần phông nền thừa nằm bên trong hình chữ nhật của bounding box.
3. **`instance_id` dùng để làm gì? Nó không phải class ID và không phải tracking ID — bạn hiểu câu này thế nào?**  
   - `instance_id` dùng để phân biệt các cá thể/đối tượng riêng biệt cùng xuất hiện trong một bức ảnh (ví dụ: chiếc xe thứ nhất và chiếc xe thứ hai).
   - Nó **không phải class ID** vì class ID dùng để chỉ định chủng loại đối tượng (cùng thuộc class "bus").
   - Nó **không phải tracking ID** vì mã định danh này chỉ có giá trị cục bộ trong một ảnh tĩnh đơn lẻ, không liên kết đối tượng xuyên suốt qua chuỗi khung hình video liên tiếp.
4. **Viết một quy tắc vẽ biên mask để người gán nhãn làm theo.**  
   *Đường polygon phải khép kín, bám sát mép viền thực tế của vật thể với dung sai không vượt quá $\pm 2$ pixel; tại các đường cong phải tăng mật độ điểm để đảm bảo đường bao mượt mà; bắt buộc khoét rỗng (interior ring/hole) tại các khoảng trống nhìn xuyên qua được.*
5. **Trên ảnh `kitchen`, chỉ ra một chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?**  
   - *Chỗ gặp vấn đề:* Khu vực chảo, xoong kim loại màu tối treo chồng lấn lên nhau trên giá treo phía trên bếp.
   - *Lựa chọn & Giải thích:* Tôi chọn **Escalation** (hỏi ý kiến Lab Coach/chuyên gia). Do các chảo kim loại cùng màu đen sẫm, tương phản ánh sáng kém và chồng lớp phức tạp, mắt thường không thể phân định chính xác ranh giới tách rời giữa các cá thể nếu không có quy ước cụ thể từ người quản lý dự án.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| **Phân loại ảnh** | Nhãn danh mục đơn toàn ảnh (`image_id`, `class_id`, `class_name`). | Ảnh `traffic` có nhiều loại phương tiện nhưng model chỉ dự đoán `cab` với score thấp (~0.51). | Đọc guideline, xác định đối tượng chiếm diện tích lớn nhất/chủ thể chính và gán 1 nhãn duy nhất. | Kiểm tra xem nhãn được chọn có tuân thủ đúng quy tắc ưu tiên chủ thể trong guideline hay không. |
| **Phát hiện vật thể** | Tọa độ hộp `[x1, y1, x2, y2]` kèm `class_id` cho từng vật thể. | Bếp nướng (`oven`) bị dự đoán đè nhiều box trùng nhau, hoặc nhầm cánh tay ở mép ảnh thành `person 0.61`. | Dùng chuột kéo bounding box bao khít từng đối tượng, phân loại đúng taxonomy. | Soát lỗi bao lỏng (loose box), bỏ sót vật thể (False Negatives) và xóa box trùng lặp (chưa NMS). |
| **Instance segmentation** | Đa giác khép kín (`polygon_xy`, `class_name`, `instance_id`). | Xoong chảo treo chồng chéo có viền dính liền, khó nhận diện tách biệt từng instance. | Vẽ các điểm polygon viền khít theo hình khối thực tế của từng cá thể đối tượng. | Kiểm tra độ lẹm biên, độ khít của mask so với vật thật và việc khoét rỗng các khoảng nền xuyên qua. |

## 5. An toàn dữ liệu

1. **Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)**  
   Tuyệt đối tuân thủ bảo mật, không phát tán dữ liệu học tập ra ngoài phạm vi môn học và không đưa bất kỳ thông tin định danh cá nhân (PII) nào vào báo cáo, mã nguồn và kho lưu trữ công khai.
2. **Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?**  
   Tôi sẽ dừng thao tác ngay lập tức, không lưu trữ hoặc chia sẻ dữ liệu đó và thông báo trực tiếp cho **Lab Coach / Giảng viên hướng dẫn**.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] `REPORT.md` (file này đã điền)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [x] Đã nộp **link repository** trên LMS