from burp import IBurpExtender

class BurpExtender(IBurpExtender):
    
    def processProxyMessage(self,messageReference, messageIsRequest, remoteHost, remotePort,
                            serviceIsHttps, httpMethod, url, resourceType, statusCode,
                            responseContentType, message, interceptAction):
        if not messageIsRequest:
            message = message.tostring().replace("java","python")
        return message

