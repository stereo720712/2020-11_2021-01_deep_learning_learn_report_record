(window.webpackJsonp=window.webpackJsonp||[]).push([[3],{1626:function(e,t,n){},1627:function(e,t,n){},1628:function(e,t,n){},1629:function(e,t,n){},1630:function(e,t,n){},1631:function(e,t,n){},1632:function(e,t,n){},1633:function(e,t,n){},1634:function(e,t,n){},1640:function(e,t,n){"use strict";n.r(t);var a=n(5),i=n(9),o=n(7),r=n(8),c=n(3),l=n(1),s=n.n(l),u=n(2),b=n.n(u),d=n(6),p=n.n(d),f=(n(1626),n(0));function v(e){var t=e.className,n=e.commercial,a=e.thanksForInviting,i=e.positive,o=e.bluebook,r=e.annotation,c=e.disclaimer,l=e.adPromotion,s=e.campaign;if(!(n||a||i||o||r||c||l||s))return null;var u=[n||a||l||o,i,r||c,s].filter(Boolean).slice(0,2);return Object(f.b)("div",{className:p()("Labels",t)},u.map((function(e,t){return Object(f.b)("div",{className:"Labels-item",key:t},e)})))}v.propTypes={className:b.a.string,commercial:b.a.node,thanksForInviting:b.a.node,bluebook:b.a.node,positive:b.a.node,annotation:b.a.node,disclaimer:b.a.node,adPromotion:b.a.node,campaign:b.a.node};var m=s.a.memo(v),h=n(473),j=n(15),O=n(96),k=n(471),y=function(e){var t=arguments.length>1&&void 0!==arguments[1]?arguments[1]:null;return[k.e[e],t].filter(Boolean).join(" · ")},L=function(e,t){switch(t){case"online_roundtable":return e.onlineRoundtable?{icon:k.b[t],title:y(t,e.onlineRoundtable.name),shortTitle:k.d[t],description:k.a[t],link:"https://www.zhihu.com/roundtable/".concat(e.onlineRoundtable.urlToken),linkDescription:k.c[t],zaName:k.g[t]}:null;case"relevant":return e.relevantInfo&&e.relevantInfo.isRelevant?{icon:k.b[t],title:y(t),shortTitle:k.d[t],description:k.a[t],zaName:k.g[t]}:null;case"professional":return e.recognitionInfo&&e.recognitionInfo.recognizedCount?{icon:k.b[t],title:y(t),shortTitle:k.d[t],description:k.a[t].replace("{}",String(e.recognitionInfo.recognizedCount)),link:"javascript:;",members:e.recognitionInfo.latestRecognizers,zaName:k.g[t]}:null;case"zhizhi_plan":var n=e.includedInfos?e.includedInfos.find((function(e){return e.commonType===t&&!k.f.includes(e.type)})):null;return n?{icon:k.b[n.type]||k.b[t],title:k.e[n.type]?y(n.type,n.title):y(t,n.title),shortTitle:k.d[n.type]||k.d[t],description:k.a[n.type]||k.a[t],link:n.url,linkDescription:k.c[n.type]||k.c[t],zaName:k.g[n.type]||n.type}:null;default:var a=e.includedInfos?e.includedInfos.find((function(e){return e.type===t})):null;return a?{icon:k.b[t],title:y(t,a.title),shortTitle:k.d[t],description:k.a[t],link:a.url,linkDescription:k.c[t],zaName:k.g[t]}:null}},R=n(97),g=n.n(R),P=n(459),N=n(10);n(1627);function B(e){var t=function(){if("undefined"==typeof Reflect||!Reflect.construct)return!1;if(Reflect.construct.sham)return!1;if("function"==typeof Proxy)return!0;try{return Date.prototype.toString.call(Reflect.construct(Date,[],(function(){}))),!0}catch(e){return!1}}();return function(){var n,a=Object(c.a)(e);if(t){var i=Object(c.a)(this).constructor;n=Reflect.construct(a,arguments,i)}else n=a.apply(this,arguments);return Object(r.a)(this,n)}}var C=function(e){Object(o.a)(n,e);var t=B(n);function n(){var e;Object(a.a)(this,n);for(var i=arguments.length,o=new Array(i),r=0;r<i;r++)o[r]=arguments[r];return(e=t.call.apply(t,[this].concat(o))).state={expanded:!1,width:null},e.bar=null,e.list=null,e.popover=null,e.trackListCardShow=g()((function(){N.a.setModule(e.list,{module:"Content",module_name:"正向列表卡片"}),N.a.trackCardShow(e.list,{id:3770})})),e.expand=function(){var t=e.props,n=t.isMobile,a=t.onExpand,i=t.data;n&&e.setState({width:e.bar.getBoundingClientRect().width}),e.setState({expanded:!0},(function(){e.trackListCardShow(),a&&a(i)}))},e.collapse=function(){e.setState({expanded:!1})},e.handleToggle=function(){e.state.expanded?e.collapse():e.expand()},e.handleDismiss=function(t){e.state.expanded&&(e.bar.contains(t.target)||e.popover.contains(t.target)||e.collapse())},e}return Object(i.a)(n,[{key:"componentDidMount",value:function(){document.addEventListener("mousedown",this.handleDismiss),document.addEventListener("touchstart",this.handleDismiss)}},{key:"componentWillUnmount",value:function(){document.removeEventListener("mousedown",this.handleDismiss),document.removeEventListener("touchstart",this.handleDismiss)}},{key:"render",value:function(){var e=this,t=this.props,n=t.data,a=t.isExpandable,i=t.renderBar,o=t.renderItems,r=t.placement,c=t.borderStyle,l=t.arrowed,s=a(n);return Object(f.b)("div",{className:"PositiveLabelLayout"},Object(f.b)("div",{ref:function(t){return e.bar=t},className:"PositiveLabelLayout-bar"},i(n,{expandable:s,onExpand:this.handleToggle})),s&&Object(f.b)(P.a,{onRef:function(t){return e.popover=t},className:p()("PositiveLabelLayout-popover","PositiveLabelLayout-popover--".concat(r),"PositiveLabelLayout-popover--".concat(c)),isOpen:this.state.expanded,target:this.bar,placement:r,arrowed:l,listenResize:!0},Object(f.b)("div",{ref:function(t){return e.list=t},className:"PositiveLabelLayout-list",style:{width:this.state.width}},o(n))))}}]),n}(s.a.Component);C.propTypes={data:b.a.object.isRequired,isExpandable:b.a.func.isRequired,renderBar:b.a.func.isRequired,renderItems:b.a.func.isRequired,placement:b.a.string,isMobile:b.a.bool.isRequired,borderStyle:b.a.oneOf(["none","common"]),arrowed:b.a.bool.isRequired,onExpand:b.a.func},C.defaultProps={borderStyle:"common"};var x=C,I=n(67),w=n(684),M=s.a.forwardRef((function(e,t){return Object(f.b)(O.a,Object(j.default)({as:"a",zaType:"Button",zaAction:"OpenUrl",extra:{link:{url:e.href}},ref:t},e))})),z=n(75);n(1628);function D(e){var t=function(){if("undefined"==typeof Reflect||!Reflect.construct)return!1;if(Reflect.construct.sham)return!1;if("function"==typeof Proxy)return!0;try{return Date.prototype.toString.call(Reflect.construct(Date,[],(function(){}))),!0}catch(e){return!1}}();return function(){var n,a=Object(c.a)(e);if(t){var i=Object(c.a)(this).constructor;n=Reflect.construct(a,arguments,i)}else n=a.apply(this,arguments);return Object(r.a)(this,n)}}var T=function(e){Object(o.a)(n,e);var t=D(n);function n(){return Object(a.a)(this,n),t.apply(this,arguments)}return Object(i.a)(n,[{key:"render",value:function(){var e=this.props.members;return Object(f.b)("div",{className:"PositiveLabelMembers"},e.slice(0,3).reverse().map((function(e){return Object(f.b)(z.a,{key:e.id,className:"PositiveLabelMembers-avatar",url:e.avatarUrl,size:20})})))}}]),n}(s.a.Component);T.propTypes={members:b.a.array};var S=T;n(1629);function E(e){var t=function(){if("undefined"==typeof Reflect||!Reflect.construct)return!1;if(Reflect.construct.sham)return!1;if("function"==typeof Proxy)return!0;try{return Date.prototype.toString.call(Reflect.construct(Date,[],(function(){}))),!0}catch(e){return!1}}();return function(){var n,a=Object(c.a)(e);if(t){var i=Object(c.a)(this).constructor;n=Reflect.construct(a,arguments,i)}else n=a.apply(this,arguments);return Object(r.a)(this,n)}}var q=function(e){Object(o.a)(n,e);var t=E(n);function n(){var e;Object(a.a)(this,n);for(var i=arguments.length,o=new Array(i),r=0;r<i;r++)o[r]=arguments[r];return(e=t.call.apply(t,[this].concat(o))).root=null,e.handleLinkClick=function(t){var n=e.props,a=n.type,i=n.link,o=n.onLinkClick,r=n.zaName;N.a.trackEvent(e.root,{id:3771,name:"一个"},{button:{text:r}}),o&&o(t,i,a)},e}return Object(i.a)(n,[{key:"componentDidMount",value:function(){var e=this,t=this.props,n=t.expandable,a=t.zaName;N.a.setModule(this.root,{module:"Content",module_name:a}),Object(I.onViewedOnce)(this.root,(function(){N.a.trackCardShow(e.root,{id:3769,name:n?"多个":"一个"})}))}},{key:"renderDesktopContent",value:function(){var e=this.props,t=e.icon,n=e.title,a=e.description,i=e.meta,o=e.link,r=e.linkDescription,c=e.members;return Object(f.b)("div",{className:"PositiveLabelBar-content"},Object(f.b)(t,{className:"PositiveLabelBar-icon"}),Object(f.b)("div",{className:"PositiveLabelBar-main"},Object(f.b)("span",{className:"PositiveLabelBar-title"},n),a&&Object(f.b)("span",{className:"PositiveLabelBar-description"}," "+a)),(o||i||c&&Boolean(c.length))&&Object(f.b)("div",{className:"PositiveLabelBar-side"},c&&Boolean(c.length)&&Object(f.b)("div",{className:"PositiveLabelBar-members"},Object(f.b)(S,{members:c})),o&&!i&&Object(f.b)("span",{className:"PositiveLabelBar-link"},r&&Object(f.b)("span",{className:"PositiveLabelBar-linkDescription"},r),Object(f.b)(w.a,{className:"PositiveLabelBar-linkIcon"})),i&&Object(f.b)("span",{className:"PositiveLabelBar-meta"},i)))}},{key:"renderMobileContent",value:function(){var e=this.props,t=e.icon,n=e.title,a=e.shortTitle,i=e.description,o=e.meta,r=e.link,c=e.expandable,l=e.members;return Object(f.b)("div",{className:"PositiveLabelBar-content"},Object(f.b)(t,{className:"PositiveLabelBar-icon"}),Object(f.b)("div",{className:"PositiveLabelBar-main"},Object(f.b)("span",{className:"PositiveLabelBar-title"},o&&a||n),i&&!o&&Object(f.b)("span",{className:"PositiveLabelBar-description"}," "+i),o&&Object(f.b)("span",{className:"PositiveLabelBar-meta"}," "+o)),(r||c||l&&Boolean(l.length))&&Object(f.b)("div",{className:"PositiveLabelBar-side"},l&&Boolean(l.length)&&Object(f.b)("div",{className:"PositiveLabelBar-members"},Object(f.b)(S,{members:l})),r&&!c&&Object(f.b)("span",{className:"PositiveLabelBar-link"},Object(f.b)(w.a,{className:"PositiveLabelBar-linkIcon"})),c&&Object(f.b)("span",{className:"PositiveLabelBar-expand"},"点击查看")))}},{key:"render",value:function(){var e=this,t=this.props,n=t.isMobile,a=t.link,i=t.expandable,o=t.onExpand,r=t.type,c=n?this.renderMobileContent():this.renderDesktopContent();return a&&!i?Object(f.b)(M,{zaText:!0,zaBlock:r,shouldTrackShow:!0,ref:function(t){return e.root=t},className:p()("PositiveLabelBar","PositiveLabelBar--link","PositiveLabelBar--".concat(r)),href:a,onClick:this.handleLinkClick},c):i?Object(f.b)(O.e,{zaText:!0,zaType:"Block",ref:function(t){return e.root=t},className:p()("PositiveLabelBar","PositiveLabelBar--expandable","PositiveLabelBar--".concat(r)),onClick:o},c):Object(f.b)("div",{ref:function(t){return e.root=t},className:"PositiveLabelBar"},c)}}]),n}(s.a.Component);q.propTypes={isMobile:b.a.bool.isRequired,icon:b.a.func.isRequired,title:b.a.string.isRequired,shortTitle:b.a.string,description:b.a.string,meta:b.a.string,link:b.a.string,linkDescription:b.a.string,onLinkClick:b.a.func,expandable:b.a.bool.isRequired,onExpand:b.a.func,members:b.a.array,zaName:b.a.string},q.defaultProps={isMobile:!1,expandable:!1};var A=q;n(1630);function _(e){var t=function(){if("undefined"==typeof Reflect||!Reflect.construct)return!1;if(Reflect.construct.sham)return!1;if("function"==typeof Proxy)return!0;try{return Date.prototype.toString.call(Reflect.construct(Date,[],(function(){}))),!0}catch(e){return!1}}();return function(){var n,a=Object(c.a)(e);if(t){var i=Object(c.a)(this).constructor;n=Reflect.construct(a,arguments,i)}else n=a.apply(this,arguments);return Object(r.a)(this,n)}}var U=function(e){Object(o.a)(n,e);var t=_(n);function n(){var e;Object(a.a)(this,n);for(var i=arguments.length,o=new Array(i),r=0;r<i;r++)o[r]=arguments[r];return(e=t.call.apply(t,[this].concat(o))).root=null,e.handleLinkClick=function(t){var n=e.props,a=n.type,i=n.link,o=n.onLinkClick,r=n.zaName;N.a.trackEvent(e.root,{id:3771,name:"一个"},{button:{text:r}}),o&&o(t,i,a)},e}return Object(i.a)(n,[{key:"componentDidMount",value:function(){var e=this,t=this.props,n=t.expandable,a=t.zaName;N.a.setModule(this.root,{module:"Content",module_name:a}),Object(I.onViewedOnce)(this.root,(function(){N.a.trackCardShow(e.root,{id:3769,name:n?"多个":"一个"})}))}},{key:"renderDesktopContent",value:function(){var e=this.props,t=e.icon,n=e.title,a=e.description,i=e.linkDescription;return Object(f.b)("div",{className:"PositiveLabelRoundtableBar-content"},Object(f.b)(t,{className:"PositiveLabelRoundtableBar-icon"}),Object(f.b)("div",{className:"PositiveLabelRoundtableBar-main"},Object(f.b)("div",{className:"PositiveLabelRoundtableBar-header"},Object(f.b)("span",{className:"PositiveLabelRoundtableBar-title"},"".concat(n," · 进行中"))),Boolean(a)&&Object(f.b)("div",{className:"PositiveLabelRoundtableBar-description"},a)),Object(f.b)("div",{className:"PositiveLabelRoundtableBar-side"},Object(f.b)("span",{className:"PositiveLabelRoundtableBar-link"},Boolean(i)&&Object(f.b)("span",{className:"PositiveLabelRoundtableBar-linkDescriptio"},i),Object(f.b)(w.a,{className:"PositiveLabelRoundtableBar-linkIcon"}))))}},{key:"renderMobileContent",value:function(){var e=this.props,t=e.icon,n=e.title,a=e.description;return Object(f.b)("div",{className:"PositiveLabelRoundtableBar-content"},Object(f.b)(t,{className:"PositiveLabelRoundtableBar-icon"}),Object(f.b)("div",{className:"PositiveLabelRoundtableBar-main"},Object(f.b)("div",{className:"PositiveLabelRoundtableBar-header"},Object(f.b)("span",{className:"PositiveLabelRoundtableBar-title"},"".concat(n," · 进行中"))),Object(f.b)("div",{className:"PositiveLabelRoundtableBar-description"},a)),Object(f.b)("div",{className:"PositiveLabelRoundtableBar-side"},Object(f.b)("span",{className:"PositiveLabelRoundtableBar-link"},Object(f.b)(w.a,{className:"PositiveLabelRoundtableBar-linkIcon"}))))}},{key:"render",value:function(){var e=this,t=this.props,n=t.isMobile,a=t.link,i=t.type,o=n?this.renderMobileContent():this.renderDesktopContent();return Object(f.b)(M,{zaText:!0,zaBlock:i,shouldTrackShow:!0,ref:function(t){return e.root=t},className:"PositiveLabelRoundtableBar PositiveLabelRoundtableBar--link",href:a,onClick:this.handleLinkClick},o)}}]),n}(s.a.Component);U.propTypes={isMobile:b.a.bool.isRequired,icon:b.a.func.isRequired,title:b.a.string.isRequired,description:b.a.string,meta:b.a.string,link:b.a.string,linkDescription:b.a.string,onLinkClick:b.a.func,expandable:b.a.bool.isRequired,onExpand:b.a.func,zaName:b.a.string},U.defaultProps={isMobile:!1,expandable:!1};var V=U;n(1631);function F(e){var t=function(){if("undefined"==typeof Reflect||!Reflect.construct)return!1;if(Reflect.construct.sham)return!1;if("function"==typeof Proxy)return!0;try{return Date.prototype.toString.call(Reflect.construct(Date,[],(function(){}))),!0}catch(e){return!1}}();return function(){var n,a=Object(c.a)(e);if(t){var i=Object(c.a)(this).constructor;n=Reflect.construct(a,arguments,i)}else n=a.apply(this,arguments);return Object(r.a)(this,n)}}var G=function(e){Object(o.a)(n,e);var t=F(n);function n(){var e;Object(a.a)(this,n);for(var i=arguments.length,o=new Array(i),r=0;r<i;r++)o[r]=arguments[r];return(e=t.call.apply(t,[this].concat(o))).root=null,e.handleLinkClick=function(t){var n=e.props,a=n.type,i=n.link,o=n.onLinkClick,r=n.zaName;N.a.trackEvent(e.root,{id:3771,name:"多个"},{button:{text:r}}),o&&o(t,i,a)},e}return Object(i.a)(n,[{key:"renderContent",value:function(){var e=this.props,t=e.icon,n=e.title,a=e.description,i=e.link,o=e.members;return Object(f.b)("div",{className:"PositiveLabelItem-content"},Object(f.b)(t,{className:"PositiveLabelItem-icon"}),Object(f.b)("div",{className:"PositiveLabelItem-main"},Object(f.b)("span",{className:"PositiveLabelItem-title"},n),a&&Object(f.b)("span",{className:"PositiveLabelItem-description"}," "+a)),(i||o&&Boolean(o.length))&&Object(f.b)("div",{className:"PositiveLabelItem-side"},o&&Boolean(o.length)&&Object(f.b)("div",{className:"PositiveLabelItem-members"},Object(f.b)(S,{members:o})),i&&Object(f.b)("span",{className:"PositiveLabelItem-link"},Object(f.b)(w.a,{className:"PositiveLabelItem-linkIcon"}))))}},{key:"render",value:function(){var e=this,t=this.props,n=t.link,a=t.type,i=this.renderContent();return n?Object(f.b)(M,{zaText:!0,zaBlock:a,shouldTrackShow:!0,ref:function(t){return e.root=t},className:"PositiveLabelItem PositiveLabelItem--link",href:n,onClick:this.handleLinkClick},i):Object(f.b)("div",{ref:function(t){return e.root=t},className:"PositiveLabelItem"},i)}}]),n}(s.a.Component);G.propTypes={isMobile:b.a.bool.isRequired,icon:b.a.func.isRequired,title:b.a.string.isRequired,description:b.a.string,link:b.a.string,onLinkClick:b.a.func,members:b.a.array,zaName:b.a.string},G.defaultProps={isMobile:!1};var H=G;function J(e){var t=function(){if("undefined"==typeof Reflect||!Reflect.construct)return!1;if(Reflect.construct.sham)return!1;if("function"==typeof Proxy)return!0;try{return Date.prototype.toString.call(Reflect.construct(Date,[],(function(){}))),!0}catch(e){return!1}}();return function(){var n,a=Object(c.a)(e);if(t){var i=Object(c.a)(this).constructor;n=Reflect.construct(a,arguments,i)}else n=a.apply(this,arguments);return Object(r.a)(this,n)}}var W=function(e){Object(o.a)(n,e);var t=J(n);function n(){var e;Object(a.a)(this,n);for(var i=arguments.length,o=new Array(i),r=0;r<i;r++)o[r]=arguments[r];return(e=t.call.apply(t,[this].concat(o))).isExpandable=function(e){return e.items.length>1&&"online_roundtable"!==e.items[0].type},e.renderBar=function(t,n){var a=n.expandable,i=n.onExpand,o=e.props,r=o.isMobile,c=o.onLinkClick,l=t.items[0],s=l.type,u=l.props,b=Object.assign({isMobile:r,type:s},u,{meta:a?"relevant"===s?"及 ".concat(t.items.length-1," 项收录"):r?"等 ".concat(t.items.length," 项收录"):"共 ".concat(t.items.length," 项收录"):null,onLinkClick:c,expandable:a,onExpand:i});return"online_roundtable"===s?Object(f.b)(V,b):Object(f.b)(A,b)},e.renderItems=function(t){var n=e.props,a=n.isMobile,i=n.onLinkClick;return t.items.map((function(e){var t=e.type,n=e.props;return Object(f.b)(H,Object(j.default)({key:t,isMobile:a,type:t},n,{onLinkClick:i}))}))},e}return Object(i.a)(n,[{key:"render",value:function(){var e=this.props,t=e.isMobile,n=e.data,a=e.placement,i=e.borderStyle,o=e.arrowed,r=e.onExpand,c=function(e){var t=k.f.map((function(t){return{type:t,props:L(e,t)}})).filter((function(e){return e.props}));return t.length?{items:t}:null}(n);return c?Object(f.b)("div",{className:"PositiveLabel"},Object(f.b)(O.b,{block:"".concat(c.items.length>1?"Multiple":"Single","_IncludeLabel")},Object(f.b)(x,{arrowed:o||t,isMobile:t,placement:a||(t?"bottom":"bottom-right"),borderStyle:i||(t?"none":"common"),data:c,isExpandable:this.isExpandable,renderBar:this.renderBar,renderItems:this.renderItems,onExpand:r}))):null}}]),n}(s.a.Component);W.propTypes={isMobile:b.a.bool.isRequired,data:b.a.object.isRequired,onLinkClick:b.a.func,placement:b.a.string,borderStyle:b.a.oneOf(["none","common"]),arrowed:b.a.bool,onExpand:b.a.func},W.defaultProps={isMobile:!1};var K=W;n(1632);function Q(e){var t=function(){if("undefined"==typeof Reflect||!Reflect.construct)return!1;if(Reflect.construct.sham)return!1;if("function"==typeof Proxy)return!0;try{return Date.prototype.toString.call(Reflect.construct(Date,[],(function(){}))),!0}catch(e){return!1}}();return function(){var n,a=Object(c.a)(e);if(t){var i=Object(c.a)(this).constructor;n=Reflect.construct(a,arguments,i)}else n=a.apply(this,arguments);return Object(r.a)(this,n)}}var X=function(e){Object(o.a)(n,e);var t=Q(n);function n(){return Object(a.a)(this,n),t.apply(this,arguments)}return Object(i.a)(n,[{key:"render",value:function(){var e=this.props,t=e.isMobile,n=e.disclaimerInfo;return n?Object(f.b)("div",{className:p()("DisclaimerLabel",{"DisclaimerLabel--mobile":t})},Object(f.b)("div",{className:"DisclaimerLabel-title"},n.description),Object(f.b)("div",{className:"DisclaimerLabel-description"},n.subDescription)):null}}]),n}(s.a.Component);X.propTypes={isMobile:b.a.bool.isRequired,disclaimerInfo:b.a.object.isRequired},X.defaultProps={isMobile:!1};var Y=X,Z=(n(330),n(93),n(23)),$=(n(90),n(309));var ee=n(1313),te=function(){return Object(f.b)(ee.a,{center:!0,size:20,css:function(e){return{fill:e.colorMod(e.colors.GRD07A).alpha(.72)}}})};function ne(e){var t=e.data,n=Object(Z.a)(e,["data"]),a=t.name,i=t.iconUrl,o=t.linkText,r=t.link;return Object(f.b)($.a,Object(j.default)({},n,{name:a,iconUrl:i,icon:Object(f.b)(te,null),linkText:o,link:r,css:function(e){return{backgroundColor:e.colorMod(e.colors.GRD07A).alpha(.08),color:e.colorMod(e.colors.GRD07A).alpha(.72)}}}))}var ae=n(17),ie=function(e){return s.a.createElement(ae.a,e,s.a.createElement("path",{d:"M15.333 19.333l-2.764 2.765a.333.333 0 0 1-.569-.236v-2.529h3.333zm0-13.333c.737 0 1.334.597 1.334 1.333V16A2.667 2.667 0 0 1 14 18.667H7.333A1.333 1.333 0 0 1 6 17.333v-10C6 6.597 6.597 6 7.333 6zm.394-1.987a2.933 2.933 0 0 1 2.93 2.791l.004.142v8.838c0 .693-.53 1.262-1.205 1.327l-.129.006V6.947a1.6 1.6 0 0 0-1.49-1.597l-.11-.004H7.333c0-.736.597-1.333 1.334-1.333h7.06z",fillRule:"evenodd"}))};ie.defaultProps={name:"BlueBook"};var oe=ie,re=function(){return Object(f.b)(oe,{center:!0,size:20})};function ce(e){var t=e.data,n=Object(Z.a)(e,["data"]),a=t.url,i=t.title;return Object(f.b)($.a,Object(j.default)({},n,{name:"知乎蓝宝书收录 · ".concat(i),icon:Object(f.b)(re,null),linkText:"前往知乎蓝宝书",link:a}))}var le=n(66),se=n(12),ue=n(136),be=n(46),de=n(65),pe=n(589),fe=n(157);n(1633);function ve(e){var t=function(){if("undefined"==typeof Reflect||!Reflect.construct)return!1;if(Reflect.construct.sham)return!1;if("function"==typeof Proxy)return!0;try{return Date.prototype.toString.call(Reflect.construct(Date,[],(function(){}))),!0}catch(e){return!1}}();return function(){var n,a=Object(c.a)(e);if(t){var i=Object(c.a)(this).constructor;n=Reflect.construct(a,arguments,i)}else n=a.apply(this,arguments);return Object(r.a)(this,n)}}var me=function(e){Object(o.a)(n,e);var t=ve(n);function n(){return Object(a.a)(this,n),t.apply(this,arguments)}return Object(i.a)(n,[{key:"render",value:function(){var e=this.props,t=e.onClose,n=e.shown,a=e.data,i=e.paging,o=(i=void 0===i?{}:i).isEnd,r=void 0!==o&&o,c=i.isLoading,l=void 0!==c&&c,s=i.totals,u=e.loadAnswerRecognizerList,b=e.answerId;return Object(f.b)(be.b,{size:"fullPage",onClose:t},n&&Object(f.b)("div",{className:"RecognizerListModal"},Object(f.b)("div",{className:"RecognizerListModal-header"},"已有 ",s||"-"," 位用户对该回答送出了专业徽章",Object(f.b)("div",{className:"RecognizerListModal-headerDescription"},"社区内具有专业创作水平的用户可以对其他创作用户送出「专业徽章」，用于认可回答的专业性，以示对其他用户创作专业内容的鼓励")),Object(f.b)(de.b,{isDrained:r,isLoading:l,onLoad:function(){u(b)}},a.map((function(e){return Object(f.b)(pe.a,{className:"UserItem",key:e.id,user:e})})))))}}]),n}(s.a.Component);me.propTypes={onClose:b.a.func.isRequired,shown:b.a.bool.isRequired,answerId:b.a.number.isRequired};var he,je=Object(se.connect)((function(e,t){var n=Object(ue.a)(e,t.answerId)||{},a=n.data,i=void 0===a?[]:a,o=n.paging,r=void 0===o?{}:o;return{data:i.map((function(t){return Object(fe.c)(e,{urlToken:t})})),paging:r}}),{loadAnswerRecognizerList:ue.d})(me);n(1634);function Oe(e){var t=function(){if("undefined"==typeof Reflect||!Reflect.construct)return!1;if(Reflect.construct.sham)return!1;if("function"==typeof Proxy)return!0;try{return Date.prototype.toString.call(Reflect.construct(Date,[],(function(){}))),!0}catch(e){return!1}}();return function(){var n,a=Object(c.a)(e);if(t){var i=Object(c.a)(this).constructor;n=Reflect.construct(a,arguments,i)}else n=a.apply(this,arguments);return Object(r.a)(this,n)}}var ke=function(e){var t=e.data;return Object(f.b)(ne,{data:Object(l.useMemo)((function(){var e,n;return{name:t.verb,link:t.targetLink,linkText:t.entryVerb,iconUrl:null===(e=t.pictures)||void 0===e||null===(n=e[0])||void 0===n?void 0:n.url}}),[t])})},ye=Object(le.a)()(he=function(e){Object(o.a)(n,e);var t=Oe(n);function n(){var e;Object(a.a)(this,n);for(var i=arguments.length,o=new Array(i),r=0;r<i;r++)o[r]=arguments[r];return(e=t.call.apply(t,[this].concat(o))).state={recognizerListModalShown:!1},e.handlePositiveLabelClick=function(t,n,a){"professional"===a&&e.toggleRecognizerModal()},e.toggleRecognizerModal=function(){e.setState({recognizerListModalShown:!e.state.recognizerListModalShown})},e}return Object(i.a)(n,[{key:"render",value:function(){var e,t,n=this.props,a=n.data,i=n.item,o=n.commercialLabels,r=n.annotationLabels,c=n.editing,l=this.state.recognizerListModalShown,u=i.isLabeled,b=i.type,d=null;a?d=(Boolean(null===(t=a.includedInfos)||void 0===t?void 0:t.length)||a.onlineRoundtable||a.relevantInfo||a.recognitionInfo)&&Object(f.b)(K,{item:i,isMobile:void 0,data:a,onLinkClick:this.handlePositiveLabelClick}):u&&(d=Object(f.b)("div",{className:"LabelContainer-placeholder"}));var p=u&&!c&&!o&&(null==a?void 0:a.thankInviterInfo)&&Object(f.b)(h.a,{inviter:a.thankInviterInfo}),v="answer"===b&&(null==a||null===(e=a.bannerInfo)||void 0===e?void 0:e[0]);return Object(f.b)(s.a.Fragment,null,Object(f.b)(m,{className:"LabelContainer",thanksForInviting:p,positive:d,annotation:r,commercial:o,disclaimer:Boolean(null==a?void 0:a.disclaimerInfo)&&Object(f.b)(Y,{disclaimerInfo:a.disclaimerInfo}),campaign:Boolean(null==v?void 0:v.verb)&&Object(f.b)(ke,{data:v}),bluebook:Boolean(null==a?void 0:a.bluebookInfo)&&Object(f.b)(ce,{data:a.bluebookInfo})}),Object(f.b)(je,{onClose:this.toggleRecognizerModal,answerId:Number(i.id),shown:l}))}}]),n}(s.a.PureComponent))||he;ye.propTypes={data:b.a.object,item:b.a.object.isRequired,commercialLabels:b.a.node,annotationLabels:b.a.node};t.default=ye}}]);
//# sourceMappingURL=column.Labels.90934105c141c43a5457.js.map