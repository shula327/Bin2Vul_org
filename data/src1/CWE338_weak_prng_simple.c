// 弱随机数生成器 (CWE-338)
char* generate_session_id_unsafe() {
    // 使用time作为随机数种子，这是可预测的
    srand(time(NULL));
    
    char* session_id = (char*)malloc(17); // 16个字符 + 终止符
    
    // 使用rand()生成随机数，这是一个弱伪随机数生成器
    for (int i = 0; i < 16; i++) {
        int r = rand() % 16;
        sprintf(&session_id[i], "%x", r);
    }
    
    session_id[16] = '\0';
    return session_id;
} 