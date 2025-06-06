import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app

def handler(event, context):
    # Netlify function wrapper
    import json
    from werkzeug.wrappers import Request, Response
    from werkzeug.serving import WSGIRequestHandler
    
    # Convert Netlify event to WSGI environ
    environ = {
        'REQUEST_METHOD': event.get('httpMethod', 'GET'),
        'PATH_INFO': event.get('path', '/'),
        'QUERY_STRING': event.get('queryStringParameters', ''),
        'CONTENT_TYPE': event.get('headers', {}).get('content-type', ''),
        'CONTENT_LENGTH': str(len(event.get('body', ''))),
        'wsgi.input': event.get('body', ''),
        'wsgi.url_scheme': 'https',
        'SERVER_NAME': 'localhost',
        'SERVER_PORT': '443',
    }
    
    # Add headers
    for key, value in event.get('headers', {}).items():
        key = key.upper().replace('-', '_')
        if key not in ('CONTENT_TYPE', 'CONTENT_LENGTH'):
            environ[f'HTTP_{key}'] = value
    
    response = app(environ, lambda status, headers: None)
    
    return {
        'statusCode': 200,
        'headers': {'Content-Type': 'text/html'},
        'body': ''.join(response)
    }