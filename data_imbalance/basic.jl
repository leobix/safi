#Calculate the AUC and F1-score for logisic regression
function getmetric(X,y,beta,beta0)
    num = size(y)[1]
    y_pred = zeros(num)
    for i = 1:num
        val = beta'*Matrix(X)[i,:]+beta0
        y_pred[i] = exp(val)/(1+exp(val))
    end

    y_py = Array(y)+ones(size(y))
    fpr, tpr, thresholds = sk.metrics.roc_curve(y_py,y_pred, pos_label=2)

    sol_res_out = zeros(2)
    sol_res_out[1] = round(sk.metrics.auc(fpr, tpr),digits=4)
    sol_res_out[2] = round(sk.metrics.f1_score(y,round.(y_pred)),digits=4)
    return sol_res_out
end

#Calculate the AUC and F1-score for SVM
function getmetric_svm(X,y,beta,beta0)
    num = size(y)[1]
    y_pred = zeros(num)
    for i = 1:num
        y_pred[i] = beta'*Matrix(X)[i,:]-beta0
    end

    y_pred2 = zeros(num)
    for i = 1:num
        if y_pred[i] > 0
            y_pred2[i] = 1
        else
            y_pred2[i] = 0
        end
    end

    y_py = Array(y)+ones(size(y))
    fpr, tpr, thresholds = sk.metrics.roc_curve(y_py,y_pred, pos_label=2)

    sol_res_out = zeros(2)
    sol_res_out[1] = round(sk.metrics.auc(fpr, tpr),digits=4)
    sol_res_out[2] = round(sk.metrics.f1_score(y,y_pred2),digits=4)

    return sol_res_out
end

function robustlogreg(train_0,train_1,coef0,coef1,class1_weight)

    model = Model(with_optimizer(Ipopt.Optimizer))
    n0,np = size(train_0)
    n1 = size(train_1)[1]
    lambda = 1

    @variable(model, theta0)
    @variable(model, theta1)
    @variable(model, p[1:n0] <=0)
    @variable(model, q[1:n1] <=0)
    @variable(model, b[1:np])
    @variable(model, b0)
    @variable(model, hs_norm >= 0, start = 0.0)
    @variable(model, absb[1:np] >= 0)
    @NLobjective(model, Max, coef0*theta0 + coef1*theta1+ sum(p[i] for i = 1:n0) + sum(q[j] for j = 1:n1)-lambda*hs_norm)

    @NLconstraint(model,con0[i=1:n0],theta0+p[i] <= -log(1+exp(b0+sum(b[l]*train_0[i,l] for l=1:np))))
    @NLconstraint(model,con1[j=1:n1],theta1+q[j] <= -class1_weight*log(1+exp(-b0-sum(b[l]*train_1[j,l] for l=1:np))))
    @NLconstraint(model,hs_norm == sum(b[j]*b[j] for j = 1:np))

    optimize!(model)

    beta0 = value(b0)
    beta = zeros(np)

    for i = 1:np
        beta[i] = value(b[i])
    end

    return beta,beta0
end

function robustlogreg_bound(train_0,train_1,coef0,coef1,class1_weight,b_opt,b0_opt,err)

    model = Model(with_optimizer(Ipopt.Optimizer))
    n0,np = size(train_0)
    n1 = size(train_1)[1]
    lambda = 1

    @variable(model, theta0)
    @variable(model, theta1)
    @variable(model, p[1:n0] <=0)
    @variable(model, q[1:n1] <=0)
    @variable(model, b[1:np])
    @variable(model, b0)
    @variable(model, hs_norm >= 0, start = 0.0)
    @variable(model, absb[1:np] >= 0)
    @NLobjective(model, Max, coef0*theta0 + coef1*theta1+ sum(p[i] for i = 1:n0) + sum(q[j] for j = 1:n1)-lambda*hs_norm)

    @NLconstraint(model,con0[i=1:n0],theta0+p[i] <= -log(1+exp(b0+sum(b[l]*train_0[i,l] for l=1:np))))
    @NLconstraint(model,con1[j=1:n1],theta1+q[j] <= -class1_weight*log(1+exp(-b0-sum(b[l]*train_1[j,l] for l=1:np))))
    @NLconstraint(model,hs_norm == sum(b[j]*b[j] for j = 1:np))

    @constraint(model,b0 >= b0_opt-err)
    @constraint(model,b0_opt >= b0-err)
    for i = 1:np
        @constraint(model,b[i] >= b_opt[i]-err)
        @constraint(model,b_opt[i] >=b[i]-err)
    end

    optimize!(model)

    beta0 = value(b0)
    beta = zeros(np)

    for i = 1:np
        beta[i] = value(b[i])
    end

    return beta,beta0
end

function calval(beta,beta0,valid_0,valid_1,method,ind0,ind1)
   res = 0

   if method == "all_fscore"
      valid0num,valid1num = size(valid_0)[1],size(valid_1)[1]
      valid_full = zeros(Float64,(valid0num+valid1num,2))

      for j = 1:valid1num
          val = beta'*Matrix(valid_1)[j,:]+beta0
          valid_full[j,1] = exp(val)/(1+exp(val))
      end

      for j = 1:valid0num
          val = beta'*Matrix(valid_0)[j,:]+beta0
          valid_full[j+valid1num,1] = exp(val)/(1+exp(val))
      end

      valid_full[1:valid1num,2] = ones(valid1num)
      res = round(sk.metrics.f1_score(valid_full[:,2],round.(valid_full[:,1]),average="binary"),digits=4)
   end

   if method == "balance_fscore"
      por = Int(round(size(valid_0)[1]/size(valid_1)[1]))
      res = 0

      valid_0 = valid_0[ind0,:]
      valid_1 = valid_1[ind1,:]

      valid0num,valid1num = size(valid_0)[1],size(valid_1)[1]
      valid0num_use = min(valid1num*por,valid0num)

      valid_full = zeros(Float64,(valid0num_use+valid1num,2))
      valid_temp = zeros(Float64,(valid0num,2))
      for j = 1:valid1num
          val = beta'*Matrix(valid_1)[j,:]+beta0
          valid_full[j,1] = exp(val)/(1+exp(val))
      end

      for j = 1:valid0num
          val = beta'*Matrix(valid_0)[j,:]+beta0
          valid_temp[j,1] = exp(val)/(1+exp(val))
      end

      valid_full[valid1num+1:valid1num+valid0num_use,1] = -sort(-valid_temp[:,1])[1:valid0num_use,1]
      valid_full[1:valid1num,2] = ones(valid1num)
      res = round(sk.metrics.f1_score(valid_full[:,2],round.(valid_full[:,1]),average="binary"),digits=4)
   end

   return res
end

function getindex(beta,beta0,train_0,train_1,coef0,coef1,class1_weight)

    model2 = Model(with_optimizer(Cbc.Optimizer))

    n0,np = size(train_0)
    n1 = size(train_1)[1]

    ulog = zeros(n0)
    vlog = zeros(n1)

    for i = 1:n0
        ulog[i] = log(1+exp(beta'*Matrix(train_0)[i,:]+beta0))
    end

    for j = 1:n1
        vlog[j] = log(1+exp(-beta'*Matrix(train_1)[j,:]-beta0))
    end

    @variable(model2,u[1:n0],Bin)
    @variable(model2,v[1:n1],Bin)

    @objective(model2,Min,-sum(ulog[i]*u[i] for i=1:n0)-class1_weight*sum(vlog[j]*v[j] for j=1:n1))
    @constraint(model2,sum(u[i] for i=1:n0) == coef0)
    @constraint(model2,sum(v[j] for j=1:n1) == coef1)

    optimize!(model2)

    usol = zeros(n0)
    vsol = zeros(n1)

    for i = 1:n0
        usol[i] = value(u[i])
    end

    for i = 1:n1
        vsol[i] = value(v[i])
    end

    use0 = sortperm(-usol)[1:coef0]
    use1 = sortperm(-vsol)[1:coef1]

    notuse0 = sortperm(-usol)[coef0+1:end]
    notuse1 = sortperm(-vsol)[coef1+1:end]

    return use0,use1,notuse0,notuse1
end

function robustlr(all_k,df_all,diff,err)
    (X_all,X_train,X_valid,X_test,y_all,y_train,y_valid,y_test) =  df_all
    all_0,all_1 = X_all[y_all.==0,:],X_all[y_all.==1,:]
    all0num,all1num = size(all_0)[1],size(all_1)[1]
    k_num = size(all_k)[1]

    clf = sk.linear_model.LogisticRegression(penalty="l2",class_weight="balanced",random_state=0,C=1).fit(X_all,y_all)
    p = size(X_all)[2]
    beta_s= zeros(p)
    for i = 1:p
        beta_s[i] = clf.coef_[1,i]
    end
    beta0_s = -clf.intercept_[1]

    ci = 1
    sol_save = Dict()

    for k =1:k_num
        ck = all_k[k]
        coef1 = Int(round(train_valid_split*all1num))
        coef0 = Int(round(coef1*ck))
        beta,beta0 = robustlogreg(all_0,all_1,coef0,coef1,ck)

        if sum(abs.(beta))  < diff
            beta,beta0 = robustlogreg_bound(all_0,all_1,coef0,coef1,ck,beta_s,beta0_s,err)
        end

        sol_save[ci] = (ck,beta,beta0)
        ci += 1
    end

    all_0,all_1 = X_all[y_all.==0,:],X_all[y_all.==1,:]
    all0num,all1num = size(all_0)[1],size(all_1)[1]
    val_num = size(collect(keys(sol_save)))[1]
    valid_auc = zeros(val_num)

    for i = 1:val_num
        ck,beta,beta0 = sol_save[i]
        coef1 = Int(round(train_valid_split*all1num))
        coef0 = Int(round(coef1*ck))
        use0,use1,notuse0,notuse1 = getindex(beta,beta0,all_0,all_1,coef0,coef1,ck)
        valid_auc[i] =  calval(beta,beta0,all_0,all_1,"all_fscore",notuse0,notuse1)
    end


    best_index = argmax(convert(Array,valid_auc))
    ck = sol_save[best_index][1]
    coef1 = Int(round(all1num))
    coef0 = Int(round(coef1*ck))
    beta,beta0 = robustlogreg(all_0,all_1,coef0,coef1,ck)
    if sum(abs.(beta)) < diff
        beta,beta0 = robustlogreg_bound(all_0,all_1,coef0,coef1,ck,beta_s,beta0_s,err)
    end

    return beta,beta0
end

function robustsvm_bound(train_0,train_1,coef0,coef1,class1_weight,sol_type,b_opt,b0_opt,err,diff)

    model = Model(with_optimizer(Ipopt.Optimizer))
    n0,np = size(train_0)
    n1 = size(train_1)[1]
    lambda = 1

    @variable(model, theta0)
    @variable(model, theta1)
    @variable(model, p[1:n0] >=0)
    @variable(model, q[1:n1] >=0)
    @variable(model, b[1:np])
    @variable(model, b0)
    @variable(model, hs_norm >= 0, start = 0.0)
    @variable(model, absb[1:np] >= 0)
    @NLobjective(model, Min, coef0*theta0 + coef1*theta1+ sum(p[i] for i = 1:n0) + sum(q[j] for j = 1:n1)+lambda*hs_norm)

    @constraint(model,con0[i=1:n0],theta0+p[i] >=0)
    @constraint(model,con2[i=1:n0],theta0+p[i] >=1+sum(b[l]*train_0[i,l] for l=1:np)-b0)
    @constraint(model,con1[j=1:n1],theta1+q[j] >=0)
    @constraint(model,con3[j=1:n1],theta1+q[j] >=class1_weight*(1-sum(b[l]*train_1[j,l] for l=1:np)+b0))

    if sol_type == 1
        @constraint(model,b0 >= b0_opt-err)
        @constraint(model,b0_opt >= b0-err)
        for i = 1:np
            @constraint(model,b[i] >= b_opt[i]-err)
            @constraint(model,b_opt[i] >=b[i]-err)
        end
    end

    @NLconstraint(model,hs_norm == sum(b[j]*b[j] for j = 1:np))

    optimize!(model)

    beta0 = value(b0)
    beta = zeros(np)

    for i = 1:np
        beta[i] = value(b[i])
    end

    return beta,beta0
end

function robustsvm(all_k,df_all,diff,err)
    (X_all,X_train,X_valid,X_test,y_all,y_train,y_valid,y_test) =  df_all
    all_0,all_1 = X_all[y_all.==0,:],X_all[y_all.==1,:]
    all0num,all1num = size(all_0)[1],size(all_1)[1]
    k_num = size(all_k)[1]
    valid_auc = zeros(k_num)

    ci = 1
    sol_save = Dict()

    #Initial Solution
    clf = sk.svm.SVC(kernel="linear",class_weight="balanced",random_state=0).fit(X_all,y_all)
    p = size(X_all)[2]
    beta_s= zeros(p)
    for i = 1:p
        beta_s[i] = clf.coef_[1,i]
    end
    beta0_s = -clf.intercept_[1]

    for k =1:k_num
        ck = all_k[k]
        coef1 = Int(round(train_valid_split*all1num))
        coef0 = Int(round(coef1*ck))
        beta,beta0 = robustsvm_bound(all_0,all_1,coef0,coef1,ck,0,beta_s,beta0_s,err,diff)
        if sum(abs.(beta)) < diff
            beta,beta0 = robustsvm_bound(all_0,all_1,coef0,coef1,ck,1,beta_s,beta0_s,err,diff)
        end

        valid_auc[k] = getmetric_svm(X_all,y_all,beta,beta0)[2]
        sol_save[k] = (ck,beta,beta0)
    end


    best_index = argmax(convert(Array,valid_auc))
    ck = sol_save[best_index][1]
    coef1 = all1num
    coef0 = Int(round(coef1*ck))

    beta,beta0 = robustsvm_bound(all_0,all_1,coef0,coef1,ck,0,beta_s,beta0_s,err,diff)
    if sum(abs.(beta)) < diff
          beta,beta0 = robustsvm_bound(all_0,all_1,coef0,coef1,ck,1,beta_s,beta0_s,err,diff)
    end

    return beta,beta0
end

function getbalancedt(df_all,k,diff,err)
    all_0,all_1 = X_all[y_all.==0,:],X_all[y_all.==1,:]
    all0num,all1num = size(all_0)[1],size(all_1)[1]

    clf = sk.linear_model.LogisticRegression(penalty="l2",class_weight="balanced",random_state=0,C=1).fit(X_all,y_all)
    p = size(X_all)[2]
    beta_s= zeros(p)
    for i = 1:p
        beta_s[i] = clf.coef_[1,i]
    end
    beta0_s = -clf.intercept_[1]

    coef1 = Int(round(all1num))
    coef0 = Int(round(coef1*k))
    beta,beta0 = robustlogreg(all_0,all_1,coef0,coef1,k)
    if sum(abs.(beta)) < diff
        beta,beta0 = robustlogreg_bound(all_0,all_1,coef0,coef1,k,beta_s,beta0_s,err)
    end

    use0,use1,notuse0,notuse1 = getindex(beta,beta0,all_0,all_1,coef0,coef1,k)
    X_all_new = [all_0[use0,:];all_1[use1,:]]
    y_all_new = [zeros(size(use0)[1]);ones(size(use1)[1])]

    return [X_all_new y_all_new]
end
