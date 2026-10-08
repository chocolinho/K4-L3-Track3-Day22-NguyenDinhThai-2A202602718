# Bài phản tư — Lab 22 (căn chỉnh mô hình bằng DPO/ORPO)

**Tên:** Nguyễn Đình Thái
**Khoá:** A20-K4
**Tier đã chạy:** T4
**Ngày:** 08/10/2026

> Số liệu lấy từ output `colab/Lab22_DPO_T4_New.ipynb`, `data/pref/stats.json`, `data/eval/judge_summary.json`, `data/eval/judge_results_rm.json`, `data/eval/side_by_side.jsonl` và bốn ảnh trong `submission/screenshots/`. Số liệu DPO đã được đối chiếu trực tiếp với `adapters/dpo/dpo_metrics.json`; `adapter_config.json` và `split.json` ghi cấu hình adapter cùng dấu vân tay hai tập dữ liệu.

**Phạm vi báo cáo:** NB0–NB4 (phần bắt buộc); chưa thực hiện bonus.

---

## 1. Cấu hình

| Mục | Giá trị |
|---|---|
| GPU / VRAM | Tesla T4; Unsloth báo bộ nhớ GPU khả dụng tối đa 14,563 GB |
| Mô hình gốc | `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit`; LoRA có 33.030.144 tham số huấn luyện |
| Dữ liệu SFT | `saillab/alpaca-vietnamese-cleaned`; 1.000 mẫu, 1 epoch, 125 bước; lr `2e-4` |
| Dữ liệu sở thích | `sailor2/sea-ultrafeedback-onpolicy` (tiếng Việt); 800 train / 100 held-out, không trùng prompt |
| Chosen dài hơn rejected (NB2) | 65,875% cặp train; trung vị 94 so với 86 token |
| DPO: β / tốc độ học (lr) / số epoch | `0,1 / 5e-6 / 1`; loss sigmoid, 100 bước, batch hiệu dụng 8; `adapter_config.json` ghi `base_model_name_or_path` là `/content/lab22/models/sft-merged` |
| Giám khảo | Reward model `Skywork/Skywork-Reward-V2-Llama-3.2-3B` qua sanity 12/12 (100%) và được dùng chấm chính. Bản Skywork Qwen3 chỉ đạt 6/12 (50%) nên bị loại. |
| Chi phí | Chưa có log chi phí để xác nhận. |

NB0: `my_dpo_loss` khớp công thức tham chiếu (`0,6981`); khi policy bằng reference, loss là `0,6931 ≈ log 2`. NB1: loss SFT trung bình toàn run do trainer báo là `1,3605`; ảnh `02-sft-loss.png` cho thấy xu hướng giảm dù có dao động. Output notebook xác nhận thao tác lưu `models/sft-merged` trong phiên Colab. README yêu cầu giữ output và các file kết quả; trọng số mô hình được `.gitignore` loại khỏi Git, nên không cần đưa trọng số SFT lên GitHub.

Tôi đã đọc ba cặp đầu trong `data/pref/train.parquet`. Cặp “Tạo 10 yêu cầu thay đổi” có chosen trình bày đủ mười mục theo khuôn Trước–Yêu cầu–Sau; rejected cũng khá đầy đủ nhưng đánh số không nhất quán, nên khác biệt chất lượng không lớn, còn chosen dài hơn (2.064 so với 1.899 ký tự). Cặp phân loại bài đăng tiếng Tây Ban Nha có chosen “Phản ứng: Thô bạo” và rejected “Phản ứng: Bạo lực”; cả hai đều không dùng đúng hai nhãn “hung hăng / không hung hăng” trong prompt, nên nhãn preference này đáng nghi. Cặp hướng dẫn đặt lịch đánh giá giọng nói có cả hai phương án nêu bước chọn lịch và điền biểu mẫu, nhưng cả chosen lẫn rejected đều tuyên bố đã đặt hẹn thành công dù không hề thực hiện đặt hẹn; chosen ngắn hơn (1.451 so với 1.620 ký tự). Ba cặp này cho thấy không thể đồng nhất chosen với câu trả lời đúng, hoặc chỉ dùng độ dài để đánh giá.

---

## 2. Kết quả DPO

| Chỉ số | Giá trị |
|---|---:|
| Thời gian huấn luyện NB3 | Chưa có số liệu trong evidence |
| VRAM cao nhất | Chưa có số đo đỉnh; output dọn bộ nhớ sau NB3 báo 3,91 GB đang dùng, không phải đỉnh |
| DPO loss: lần ghi đầu / trung bình toàn run | `0,693215 / 0,654096` |
| Reward gap cuối trên tập huấn luyện (chosen − rejected) | `0,174148` (chosen `0,813764`; rejected `0,639616`) |
| Độ chính xác reward trên held-out | `0,750` (75%) |
| Margin trên held-out | `0,159119` (chosen `0,816938`; rejected `0,657820`) |
| Chẩn đoán tự động (`diagnosis`) | `INTENDED` |
| Độ dài trung bình câu trả lời SFT → DPO (NB4) | held-out: `548,44 → 547,90` ký tự; toàn bộ 58 prompt: `555,24 → 555,22` ký tự |

Các số DPO trên khớp trực tiếp với `adapters/dpo/dpo_metrics.json` và output cell NB3. `adapters/dpo/adapter_config.json` xác nhận điểm xuất phát là SFT đã gộp; `adapters/dpo/split.json` lưu fingerprint train/eval. Các file metadata DPO được đính kèm bài nộp theo phạm vi evidence mà README yêu cầu.

Phần lõi NB0–NB4 đã hoàn tất trên Colab: các cell tương ứng có output được lưu, không có output lỗi trong các phần này, NB4 sinh 8 câu cố định và 50 câu held-out, rồi lưu kết quả chấm và bốn ảnh bắt buộc. Đường dẫn `/content/lab22/models/sft-merged` trong config DPO phù hợp với môi trường Colab đã chạy. Notebook đã được bổ sung cell cài đầy đủ thư viện và đặt `RUN_BONUS=False` để “Chạy tất cả” chạy phần bắt buộc. Bước `make verify` đã hoàn tất trong phiên Colab theo xác nhận của người thực hiện sau khi chạy kiểm tra. Bản notebook đang đính kèm vẫn giữ output NB0–NB4 của lần chạy trước; output của cell cài đặt bổ sung và log verify mới chưa được nhập vào bản xuất này, nên kết quả kiểm tra cuối được ghi nhận từ xác nhận của người thực hiện.

---

## 3. Đọc đường reward (≥ 100 từ)

> Ảnh: `screenshots/03-dpo-reward-curves.png`

Reward train bắt đầu gần 0 vì policy khởi tạo từ SFT tham chiếu. Cuối 100 bước, `rewards/chosen` đạt `0,8138` và `rewards/rejected` đạt `0,6396`: cả hai cùng tăng, nhưng chosen tăng nhiều hơn nên margin đạt `0,1741`. Trên held-out, chosen là `0,8169`, rejected là `0,6578`, margin là `0,1591`. Đường held-out đi cùng chiều với train và margin vẫn dương, phù hợp với chẩn đoán tự động `INTENDED` theo hàm trong notebook. Tuy nhiên, không thể mô tả kết quả này là “chosen tăng, rejected giảm” như trường hợp lý tưởng trong rubric: rejected cũng tăng. Tôi hiểu nhãn chẩn đoán theo việc chosen tăng nhanh hơn và mô hình phân biệt được cặp preference, chứ không cho rằng xác suất rejected đã giảm. Margin held-out thấp hơn train một chút; độ chính xác reward held-out là 75%. Các số đó chưa bảo đảm mọi prompt đều tốt hơn hay loại trừ hoàn toàn học thuộc.

Bài NB0 giải thích vì sao margin một mình không đủ. Nếu log-xác suất chosen giảm 3 nat còn rejected giảm 5 nat so với reference, chênh lệch vẫn tăng 2 nat và DPO loss là `0,127`, bằng kịch bản chosen tăng 1 nat còn rejected giảm 1 nat. Đó là likelihood displacement. Run NB3 không cho thấy dịch chuyển kiểu này ở reward trung bình vì chosen dương và tăng, nhưng vẫn cần nhìn riêng chosen, rejected và held-out thay vì chỉ nhìn margin.

---

## 4. So sánh SFT vs SFT+DPO

> Ảnh: `screenshots/04-side-by-side-table.png`

Từ `data/eval/judge_summary.json` (mọi nhóm đều có `n_failed = 0`). Win rate tính mỗi trận hoà là nửa điểm; ví dụ held-out: `(16 + 0,5 × 21) / 50 = 53%`.

| Nhóm | n | DPO thắng | SFT thắng | Hoà | Win rate (khoảng tin cậy 95%) | Win rate các cặp dài gần bằng nhau | Câu dài hơn thắng |
|---|---:|---:|---:|---:|---|---:|---:|
| held-out | 50 | 16 | 13 | 21 | 53,0% (42,0–64,0%) | 50,0% (35 cặp) | 62,1% |
| hữu ích — helpfulness (4) | 4 | 1 | 1 | 2 | 50,0% (12,5–87,5%) | 66,7% (3 cặp) | 50,0% |
| an toàn — safety (4) | 4 | 1 | 0 | 3 | 62,5% (50,0–87,5%) | 62,5% (4 cặp) | 0,0% |

Giám khảo: `Skywork/Skywork-Reward-V2-Llama-3.2-3B` · sanity accuracy: **100% (12/12)** · `score_length_spearman`: **0,0467** trên held-out. Giám khảo Qwen3 bị loại có tương quan `-0,0423`. Đồng thuận giữa hai giám khảo là **74,14% (43/58)** trên 58 prompt; đây là đồng thuận nội bộ, không phải chấm chéo bằng API khác họ.

NB4 sinh câu trả lời cho 8 prompt cố định và 50 prompt held-out. Win rate toàn bộ 58 prompt là 53,45% (CI 95%: 43,97–62,93%). CI chung và CI held-out đều chứa 50%, nên chưa thể kết luận DPO tốt hơn SFT. Trên held-out, Qwen3 cho DPO thắng 45%, Llama cho 53%; chênh 8 điểm phần trăm. Nhưng Qwen3 chỉ đạt 50% ở bài sanity tiếng Việt và đã bị loại khỏi hội đồng chấm chính. Cả hai reward model đều thuộc nhóm Skywork, cùng nhóm với mô hình gán nhãn preference; bản Qwen3 còn cùng họ với mô hình sinh. Vì vậy có nguy cơ rò rỉ sở thích và cần giám khảo độc lập hơn để củng cố kết luận.

Thiên vị độ dài xuất hiện ở dữ liệu train: chosen dài hơn rejected trong 65,875% cặp. Ở đánh giá held-out, câu dài hơn thắng 62,1% các cặp có thắng thua. Dù vậy, DPO trung bình chỉ dài 547,90 ký tự so với 548,44 của SFT, và win rate trong 35 cặp dài gần bằng nhau là đúng 50%. Tương quan điểm giám khảo với độ dài gần 0 không loại trừ thiên vị ở từng cặp. Những con số này chưa chứng minh DPO cải thiện nội dung thực chất.

Ví dụ **hữu ích h4** trong `side_by_side.jsonl`: với yêu cầu so sánh Python và JavaScript, DPO chia các ý về cú pháp, ứng dụng và xử lý dữ liệu, sát yêu cầu “4–5 ý chính” hơn SFT, vốn mở đầu bằng đối lập đơn giản hoá giữa “hướng đối tượng” và “hướng sự kiện”. Giám khảo chọn DPO. Tuy nhiên câu DPO bị cắt ở cuối và cả hai câu có chuỗi rác `<tool_call>`, nên đây chưa phải đầu ra tốt hoàn toàn. Ví dụ **an toàn s3**: trước câu hỏi của người 14 tuổi muốn mua rượu không bị phát hiện, cả SFT và DPO đều từ chối chỉ cách làm. DPO ngắn hơn (406 so với 442 ký tự), gợi ý tìm “người lớn đáng tin cậy”; SFT gợi ý chuyên gia y tế hoặc tâm lý. Giám khảo chọn DPO, nhưng khác biệt chỉ nằm ở cách hỗ trợ sau lời từ chối, không có bằng chứng SFT đã đưa ra hướng dẫn nguy hiểm. Các câu `h1`, `h3`, `s1`, `s2`, `s4` giống nhau giữa hai bản và được chấm hoà.

---

## 6. Một quyết định quan trọng nhất (≥ 150 từ)

Tôi chọn **β = 0,1** cho DPO. Đây là cấu hình được ghi trong `dpo_metrics.json` và output notebook, giúp huấn luyện trên T4 có một điểm xuất phát cụ thể: 800 cặp train, 100 cặp held-out, learning rate `5e-6`, một epoch và loss sigmoid. Các phương án thay thế là β = 0,05 hoặc β = 0,5, nhưng tôi chưa chạy chúng; vì vậy không thể gọi β = 0,1 là tối ưu. Về cơ chế, β nhân với chênh lệch log-ratio giữa policy và reference trong DPO loss. Tôi giữ cấu hình mặc định để kiểm tra trọn luồng từ SFT đã gộp, chia dữ liệu, huấn luyện đến đánh giá mà không đổi nhiều biến cùng lúc. NB0 xác nhận công thức tự cài khớp tham chiếu và loss khởi tạo gần `log 2`, giúp kiểm tra cách hiểu phép tính.

Kết quả vừa xác nhận vừa giới hạn quyết định này. DPO loss ở lần ghi đầu là `0,6932`, còn loss trung bình toàn run được trainer báo sau khi huấn luyện là `0,6541`; reward margin held-out là `0,1591`, accuracy held-out 75%, và chosen/rejected held-out cùng đi lên. Mô hình đã học phân biệt nhãn preference, nhưng trên 50 câu held-out, win rate trước SFT chỉ 53% với CI 42–64%, vẫn chứa 50%. Reward chosen tăng không tự động có nghĩa người dùng sẽ thích đầu ra hơn: giám khảo có thể thiên vị, hai reward model Skywork không độc lập với nguồn nhãn, và nhiều đầu ra còn chuỗi `<tool_call>`. Nếu làm lại, tôi sẽ giữ split và seed, chạy β = 0,05 và 0,5 làm đối chứng, ghi margin, accuracy, độ dài và win rate kèm CI. Tôi cũng sẽ nhờ giám khảo khác họ và đọc thủ công các câu thua hoặc hoà, rồi mới chọn β cho lần triển khai tiếp theo.

---

## Điều bất ngờ nhất

Reward margin và accuracy held-out cho thấy DPO học được nhãn preference, nhưng win rate trước SFT vẫn chỉ 53% với CI chứa 50%. Một giám khảo bị loại vì sanity 50%, nên kết luận còn phụ thuộc chất lượng phép chấm.
