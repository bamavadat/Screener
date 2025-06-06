import sys
import os
import json
from urllib.parse import parse_qs, unquote

# Add the root directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Import your Flask app
from app import app

def handler(event, context):
    """Netlify function handler for Flask app"""
    try:
        # Extract request details from Netlify event
        http_method = event.get('httpMethod', 'GET')
        path = event.get('path', '/')
        query_params = event.get('queryStringParameters') or {}
        headers = event.get('headers', {})
        body = event.get('body', '')

        # Remove /api prefix if present (due to redirect)
        if path.startswith('/api'):
            path = path[4:]

        # Build query string
        query_string = '&'.join([f"{k}={v}" for k, v in query_params.items()])

        # Create WSGI environ
        environ = {
            'REQUEST_METHOD': http_method,
            'PATH_INFO': path,
            'QUERY_STRING': query_string,
            'CONTENT_TYPE': headers.get('content-type', ''),
            'CONTENT_LENGTH': str(len(body)) if body else '0',
            'SERVER_NAME': 'localhost',
            'SERVER_PORT': '443',
            'wsgi.url_scheme': 'https',
            'wsgi.input': body.encode() if isinstance(body, str) else body,
            'wsgi.errors': sys.stderr,
            'wsgi.version': (1, 0),
            'wsgi.multithread': False,
            'wsgi.multiprocess': True,
            'wsgi.run_once': False
        }

        # Add HTTP headers to environ
        for key, value in headers.items():
            key = key.upper().replace('-', '_')
            if key not in ('CONTENT_TYPE', 'CONTENT_LENGTH'):
                environ[f'HTTP_{key}'] = value

        # Response capture
        response_data = {'status': None, 'headers': [], 'body': []}

        def start_response(status, response_headers, exc_info=None):
            response_data['status'] = status
            response_data['headers'] = response_headers

        # Call Flask app
        app_response = app(environ, start_response)

        # Collect response body
        for data in app_response:
            if isinstance(data, bytes):
                response_data['body'].append(data.decode('utf-8'))
            else:
                response_data['body'].append(str(data))

        # Extract status code
        status_code = int(response_data['status'].split()[0])

        # Build response headers
        response_headers = {}
        for header_name, header_value in response_data['headers']:
            response_headers[header_name] = header_value

        # Join body
        response_body = ''.join(response_data['body'])

        return {
            'statusCode': status_code,
            'headers': response_headers,
            'body': response_body
        }

    except Exception as e:
        print(f"Error in Netlify function: {str(e)}")
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({'error': f'Function error: {str(e)}'})
        }