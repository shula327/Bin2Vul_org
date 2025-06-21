
void CWE617_reachable_assert(int x) {
    // 漏洞：assert条件可控，攻击者可使其失败
    if (!(x > 0)) {
        // 断言失败
        *((int*)0) = 0;
    }
} 