// 不安全：使用已释放的内存
void use_after_free_unsafe() {
    char* ptr = (char*)malloc(10);
    strcpy(ptr, "hello");
    free(ptr);
    printf("%s\n", ptr);  // 使用已释放的内存
} 