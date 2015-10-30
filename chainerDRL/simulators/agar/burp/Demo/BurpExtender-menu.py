from burp import IBurpExtender
from burp import IMenuItemHandler

from cgi import parse_qs
import traceback, sys

class BurpExtender(IBurpExtender):
    def registerExtenderCallbacks(self, callbacks):
        self.mCallBacks = callbacks
        try:
	        self.mCallBacks.registerMenuItem("python diff", ArgsDiffMenuItem())
        except:
			print "This only works with version 1.3.07 of the suite"
            traceback.print_exc(file=sys.stdout)

class ArgsDiffMenuItem(IMenuItemHandler):
    def menuItemClicked(self, menuItemCaption, messageInfo):
        print "--- Diff on arguments ---"
        if len(messageInfo) == 2:
            # We can do a diff
            request1=HttpRequest(messageInfo[0].getRequest())
            request2=HttpRequest(messageInfo[1].getRequest())
            print "Diff in GET parameters:"
            self.diff(request1.query_params,request2.query_params)
            print "Diff in POST parameters:"
            self.diff(request1.body_params,request2.body_params)
        else:
            print "You need to select two messages to do an argument diff"
        print "\n\n"

    def diff(self, params1, params2):
            for param in params1:
                if param not in params2:
                    print "Param %s=%s is not is the second request" % \
                          (param, params1[param])
                    continue
                if params1[param] != params2[param]:
                    print "Request1 %s=%s Request2 %s=%s" % \
                            (param, params1[param], param, params2[param])
            for param in params2:
                if param not in params1:
                    print "Param %s=%s is not is the first request" % \
                          (param, params2[param])
        
class HttpRequest:
    def __init__(self, request):
        self.request=request.tostring().splitlines()
        self.query_params={}
        self.getParameters()

    def getParameters(self):
        # get url parameters
        try:
            self.query_params=parse_qs(\
            ''.join(self.request[0].split()[1].split("?")[1:]))
        except:
            self.query_params={}

        # get body parameters
        try:
            index=++self.request.index('')
            self.body_params=parse_qs(\
            ''.join(self.request[index:]))
        except:
            self.body_params={}
