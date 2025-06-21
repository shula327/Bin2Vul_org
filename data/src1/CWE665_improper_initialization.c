
void CWE665_improper_initialization(char *src) {
    char buf[20];
    // 漏洞：未初始化buf直接strcat
    strcat(buf, src);
} 