// 不安全：没有验证上传文件类型
void save_uploaded_file_unsafe(const char* filename, const char* content) {
    FILE* file = fopen(filename, "wb");
    fputs(content, file);
    fclose(file);
} 