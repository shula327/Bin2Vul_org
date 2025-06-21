void CWE123_write_what_where_condition(char *input) {
    struct node { struct node *next; struct node *prev; };
    struct node head, n1;
    head.next = &n1;
    head.prev = &n1;
    n1.next = &head;
    n1.prev = &head;
    // 漏洞：未检查input长度，直接memcpy覆盖结构体指针
    memcpy(&n1, input, strlen(input));
    printf("next: %p, prev: %p\n", n1.next, n1.prev);
} 