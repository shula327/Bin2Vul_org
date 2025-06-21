// 不安全加载库
void load_library_unsafe(const char* library_name) {
    // 库可能从不安全位置加载
    void* handle = dlopen(library_name, RTLD_LAZY);
} 