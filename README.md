# landscape-recognition

1 Tranining: 
    data thông qua model => data training
2 Nhận diện:
    model lấy data training để tìm ảnh

1 Tranining:
    sau khi tranining xong thì ta lưu: + model vào file json
                                        + ma trận weight vào file h5 ( model.save(path))
2 Test dữ liệu:
    lấy model từ file json
    lấy weight vào model (func: load_weights(path))

    trường hợp của mình là ta lấy model từ file model.py rồi nên chỉ cần load_weights() nữa thôi.