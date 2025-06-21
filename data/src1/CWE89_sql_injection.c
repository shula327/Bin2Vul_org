// 不安全：直接构建SQL查询
QueryResult execute_query_unsafe(const char* user_input) {
    char query[256];
    snprintf(query, sizeof(query), "SELECT * FROM users WHERE username='%s'", user_input);
    printf("[不安全] 执行查询: %s\n", query);
    return create_query_result(query, NULL);
} 