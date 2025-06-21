// 不安全：明文保存密码
void save_credentials_unsafe(const char* username, const char* password) {
    FILE* file = fopen("credentials.txt", "a");
    fprintf(file, "用户: %s, 密码: %s\n", username, password);
    fclose(file);
} 