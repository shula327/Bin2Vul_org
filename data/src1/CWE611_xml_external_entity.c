// 不安全：允许外部实体处理
void parse_xml_unsafe(const char* xml_data) {
    // 创建XML解析器但不禁用外部实体
    xmlDocPtr doc = xmlParseMemory(xml_data, strlen(xml_data));
} 