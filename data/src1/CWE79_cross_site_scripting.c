// 不安全：没有过滤的HTML输出
void render_comment_unsafe(const char* user_comment) {
    printf("<div class='comment'>%s</div>\n", user_comment);
    // 没有过滤特殊字符，如<script>标签，允许XSS攻击
} 