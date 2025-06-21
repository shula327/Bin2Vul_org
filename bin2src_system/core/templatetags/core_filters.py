from django import template
import re

register = template.Library()

@register.filter
def mult(value, arg):
    """
    将值乘以参数
    例如：{{ value|mult:100 }}
    """
    try:
        return float(value) * float(arg)
    except (ValueError, TypeError):
        return value

@register.filter
def cwe_extract(filename):
    """
    从文件名中提取CWE编号
    例如：{{ filename|cwe_extract }}
    支持多种格式:
    - 'CWE-123'
    - 'CWE123'
    - 'CWE121_Stack_Based_Buffer_Overflow'
    """
    # 先尝试匹配CWE-数字格式
    cwe_match = re.search(r'CWE-?(\d+)', filename)
    if cwe_match:
        return cwe_match.group(1)
    
    # 匹配开头为CWE数字的格式
    cwe_match = re.search(r'^CWE(\d+)_', filename)
    if cwe_match:
        return cwe_match.group(1)
    
    return ''