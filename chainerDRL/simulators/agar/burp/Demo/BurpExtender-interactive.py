from burp import IBurpExtender
from java.net import URL
from code import InteractiveConsole

class BurpExtender(IBurpExtender):
    def processProxyMessage(self,messageReference, messageIsRequest, remoteHost, remotePort,
                            serviceIsHttps, httpMethod, url, resourceType, statusCode,
                            responseContentType, message, interceptAction):
        if not messageIsRequest:
            uUrl = URL("HTTPS" if serviceIsHttps else "HTTP", remoteHost, remotePort, url)
            if self.mCallBacks.isInScope(uUrl):
                message = message.tostring()
                from pprint import pprint
                loc=dict(locals())
                c = InteractiveConsole(locals=loc)
                c.interact("Interactive python interpreter")
                for key in loc:
                    if key != '__builtins__':
                        exec "%s = loc[%r]" % (key, key)
        return message

    def registerExtenderCallbacks(self, callbacks):
        self.mCallBacks = callbacks
